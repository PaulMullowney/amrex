#ifndef AMREX_PCI_H_
#define AMREX_PCI_H_

template <class FAB>
void
FabArray<FAB>::PC_local_cpu (const CPC& thecpc, FabArray<FAB> const& src,
                             int scomp, int dcomp, int ncomp, CpOp op)
{
    int N_locs = thecpc.m_LocTags->size();
    if (N_locs == 0) return;
    bool is_thread_safe = thecpc.m_threadsafe_loc;

    if (is_thread_safe)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (int i = 0; i < N_locs; ++i)
        {
            const CopyComTag& tag = (*thecpc.m_LocTags)[i];
            if (this != &src || tag.dstIndex != tag.srcIndex || tag.sbox != tag.dbox) {
                // avoid self copy or plus
                const FAB* sfab = &(src[tag.srcIndex]);
                      FAB* dfab = &(get(tag.dstIndex));
                if (op == FabArrayBase::COPY)
                {
                    dfab->template copy<RunOn::Host>(*sfab, tag.sbox, scomp, tag.dbox, dcomp, ncomp);
                }
                else
                {
                    dfab->template plus<RunOn::Host>(*sfab, tag.sbox, tag.dbox, scomp, dcomp, ncomp);
                }
            }
        }
    }
    else
    {
        LayoutData<Vector<FabCopyTag<FAB> > > loc_copy_tags(boxArray(),DistributionMap());
        for (int i = 0; i < N_locs; ++i)
        {
            const CopyComTag& tag = (*thecpc.m_LocTags)[i];
            if (this != &src || tag.dstIndex != tag.srcIndex || tag.sbox != tag.dbox) {
                loc_copy_tags[tag.dstIndex].push_back
                    ({src.fabPtr(tag.srcIndex), tag.dbox, tag.sbox.smallEnd()-tag.dbox.smallEnd()});
            }
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (MFIter mfi(*this); mfi.isValid(); ++mfi)
        {
            const auto& tags = loc_copy_tags[mfi];
            auto dfab = this->array(mfi);
            if (op == FabArrayBase::COPY)
            {
                for (auto const & tag : tags)
                {
                    auto const sfab = tag.sfab->array();
                    Dim3 offset = tag.offset.dim3();
                    amrex::LoopConcurrentOnCpu (tag.dbox, ncomp,
                    [=] (int i, int j, int k, int n) noexcept
                    {
                        dfab(i,j,k,dcomp+n) = sfab(i+offset.x,j+offset.y,k+offset.z,scomp+n);
                    });
                }
            }
            else
            {
                for (auto const & tag : tags)
                {
                    auto const sfab = tag.sfab->array();
                    Dim3 offset = tag.offset.dim3();
                    amrex::LoopConcurrentOnCpu (tag.dbox, ncomp,
                    [=] (int i, int j, int k, int n) noexcept
                    {
                        dfab(i,j,k,dcomp+n) += sfab(i+offset.x,j+offset.y,k+offset.z,scomp+n);
                    });
                }
            }
        }
    }
}

#ifndef PCI_STASHED_BUFFERS
#define PCI_STASHED_BUFFERS
#endif
//#undef PCI_STASHED_BUFFERS

#ifdef AMREX_USE_GPU
template <class FAB>
void
FabArray<FAB>::PC_local_gpu (const CPC& thecpc, FabArray<FAB> const& src,
                             int scomp, int dcomp, int ncomp, CpOp op)
{
    int N_locs = thecpc.m_LocTags->size();
    if (N_locs == 0) return;
    bool is_thread_safe = thecpc.m_threadsafe_loc;

    typedef Array4CopyTag<value_type> TagType;
    Vector<TagType> loc_copy_tags;
    loc_copy_tags.reserve(N_locs);

    Vector<BaseFab<int> > maskfabs;
    Vector<Array4Tag<int> > masks;
    if (!is_thread_safe)
    {
        if ((op == FabArrayBase::COPY && !amrex::IsStoreAtomic<value_type>::value) ||
            (op == FabArrayBase::ADD  && !amrex::HasAtomicAdd <value_type>::value))
        {
            maskfabs.resize(this->local_size());
            masks.reserve(N_locs);
        }
    }

    for (int i = 0; i < N_locs; ++i)
    {
        const CopyComTag& tag = (*thecpc.m_LocTags)[i];
        if (this != &src || tag.dstIndex != tag.srcIndex || tag.sbox != tag.dbox) {
            int li = this->localindex(tag.dstIndex);
            loc_copy_tags.push_back
                ({this->atLocalIdx(li).array(),
                     src.fabPtr(tag.srcIndex)->const_array(),
                     tag.dbox,
                     (tag.sbox.smallEnd()-tag.dbox.smallEnd()).dim3()});

            if (maskfabs.size() > 0) {
                if (!maskfabs[li].isAllocated()) {
                    maskfabs[li].resize(this->atLocalIdx(li).box());
                }
                masks.emplace_back(Array4Tag<int>{maskfabs[li].array()});
            }
        }
    }

#ifdef PCI_STASHED_BUFFERS
    bool found = false;
    PCLocalDeviceTagsNwarpsData stashed_buf = {nullptr, nullptr, 0, 0, 0, 0, 0};

    /* search backwards */
    Vector<PCLocalDeviceTagsNwarpsData>::reverse_iterator p = this->pclocal_device_tags_nwarps.rbegin();
    while (p != this->pclocal_device_tags_nwarps.rend()) {
        if (memcmp((this->pclocal_device_tags_nwarps.back()).h_buffer, loc_copy_tags.data(), (this->pclocal_device_tags_nwarps.back()).nbytes)==0) {
            stashed_buf = *p;
            found = true;
            break;
        }
        p++;
    }

    if (found && stashed_buf.ntags == 0) return;

    if (!found) {
        stashed_buf.ntags = loc_copy_tags.size();
        if (stashed_buf.ntags == 0) {
            this->pclocal_device_tags_nwarps.push_back(stashed_buf);
            return;
        }

        stashed_buf.ntotwarps = 0;
        Vector<int> nwarps;
        nwarps.reserve(stashed_buf.ntags+1);
        for (int i = 0; i < stashed_buf.ntags; ++i)
        {
            auto& tag = loc_copy_tags[i];
            nwarps.push_back(stashed_buf.ntotwarps);
            stashed_buf.ntotwarps +=  static_cast<int>((detail::get_tag_size(tag)
                                                        + Gpu::Device::warp_size-1) / Gpu::Device::warp_size);
        }
        nwarps.push_back(stashed_buf.ntotwarps);

        std::size_t sizeof_tags = stashed_buf.ntags*sizeof(TagType);
        stashed_buf.offset_nwarps = Arena::align(sizeof_tags);
        stashed_buf.sizeof_nwarps = (stashed_buf.ntags+1)*sizeof(int);
        std::size_t total_buf_size = stashed_buf.offset_nwarps + stashed_buf.sizeof_nwarps;

        stashed_buf.d_buffer = (char*)The_Arena()->alloc(total_buf_size);
        stashed_buf.h_buffer = (char*)The_Pinned_Arena()->alloc(total_buf_size);
        stashed_buf.nbytes = sizeof_tags;

        this->pclocal_device_tags_nwarps.push_back(stashed_buf);

        std::memcpy(stashed_buf.h_buffer, loc_copy_tags.data(), sizeof_tags);
        std::memcpy(stashed_buf.h_buffer+stashed_buf.offset_nwarps, nwarps.data(), stashed_buf.sizeof_nwarps);
        Gpu::htod_memcpy_async(stashed_buf.d_buffer, stashed_buf.h_buffer, total_buf_size);
    }
#endif

    if (maskfabs.size() > 0) {
        amrex::ParallelFor(masks,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, Array4Tag<int> const& msk) noexcept
        {
            msk.dfab(i,j,k) = 0;
        });
    }

    if (op == FabArrayBase::COPY)
    {
        if (is_thread_safe) {
#ifdef PCI_STASHED_BUFFERS
            detail::foo<value_type>(loc_copy_tags, scomp, dcomp, ncomp, stashed_buf.d_buffer, stashed_buf.ntotwarps,
                                    stashed_buf.ntags, stashed_buf.offset_nwarps, detail::CellStore<value_type>());
#else
            detail::fab_to_fab<value_type>(loc_copy_tags, scomp, dcomp, ncomp,
                                           detail::CellStore<value_type>());
#endif
        } else {
#ifdef PCI_STASHED_BUFFERS
            detail::foo_atomic_cpy<value_type>(loc_copy_tags, scomp, dcomp, ncomp, stashed_buf.d_buffer,
                                               stashed_buf.ntotwarps, stashed_buf.ntags, stashed_buf.offset_nwarps, masks);
#else
            detail::fab_to_fab_atomic_cpy<value_type>(loc_copy_tags, scomp, dcomp, ncomp, masks);
#endif
        }
    }
    else
    {
        if (is_thread_safe) {
#ifdef PCI_STASHED_BUFFERS
            detail::foo<value_type>(loc_copy_tags, scomp, dcomp, ncomp, stashed_buf.d_buffer, stashed_buf.ntotwarps,
                                    stashed_buf.ntags, stashed_buf.offset_nwarps, detail::CellAdd<value_type>());
#else
            detail::fab_to_fab<value_type>(loc_copy_tags, scomp, dcomp, ncomp,
                                            detail::CellAdd<value_type>());
#endif
        } else {
#ifdef PCI_STASHED_BUFFERS
            detail::foo_atomic_add<value_type>(loc_copy_tags, scomp, dcomp, ncomp, stashed_buf.d_buffer,
                                               stashed_buf.ntotwarps, stashed_buf.ntags, stashed_buf.offset_nwarps, masks);
#else
            detail::fab_to_fab_atomic_add<value_type>(loc_copy_tags, scomp, dcomp, ncomp, masks);
#endif
        }
    }
}
#endif

#endif
