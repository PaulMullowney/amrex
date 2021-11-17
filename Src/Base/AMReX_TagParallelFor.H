#ifndef AMREX_TAG_PARALLELFOR_H_
#define AMREX_TAG_PARALLELFOR_H_
#include <AMReX_Config.H>

#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_FabArray.H>
#include <AMReX_Vector.H>
#include <utility>

namespace amrex {

template <class T>
struct Array4CopyTag {
    Array4<T      > dfab;
    Array4<T const> sfab;
    Box dbox;
    Dim3 offset; // sbox.smallEnd() - dbox.smallEnd()

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <class T>
struct Array4MaskCopyTag {
    Array4<T      > dfab;
    Array4<T const> sfab;
    Array4<int    > mask;
    Box dbox;
    Dim3 offset; // sbox.smallEnd() - dbox.smallEnd()

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <class T>
struct Array4Tag {
    Array4<T> dfab;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box box () const noexcept { return Box(dfab); }
};

template <class T>
struct Array4BoxTag {
    Array4<T> dfab;
    Box       dbox;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <class T>
struct Array4BoxValTag {
    Array4<T> dfab;
    Box       dbox;
    T          val;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <class T>
struct VectorTag {
    T* p;
    int m_size;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int size () const noexcept { return m_size; }
};

#ifdef AMREX_USE_GPU

namespace detail {

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::enable_if_t<std::is_same<std::decay_t<decltype(std::declval<T>().box())>, Box>::value,
                 int>
get_tag_size (T const& tag) noexcept
{
    return static_cast<int>(tag.box().numPts());
}

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::enable_if_t<std::is_integral<std::decay_t<decltype(std::declval<T>().size())> >::value,
                 int>
get_tag_size (T const& tag) noexcept
{
    return tag.size();
}

template <typename T, typename F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::enable_if_t<std::is_same<std::decay_t<decltype(std::declval<T>().box())>, Box>::value>
tagparfor_call_f (
#ifdef AMREX_USE_DPCPP
    sycl::nd_item<1> const& item,
#endif
    int icell, T const& tag, F&& f) noexcept
{
    int ncells = tag.box().numPts();
    const auto len = amrex::length(tag.box());
    const auto lo  = amrex::lbound(tag.box());
    //int tmp1 = len.x*len.y;
    //int tmp2 = icell - k*tmp1;
    int k =  icell /   (len.x*len.y);
    int j = (icell - k*(len.x*len.y)) /   len.x;
    int i = (icell - k*(len.x*len.y)) - j*len.x;
    i += lo.x;
    j += lo.y;
    k += lo.z;
#ifdef AMREX_USE_DPCPP
    f(item, icell, ncells, i, j, k, tag);
#else
    f(      icell, ncells, i, j, k, tag);
#endif
}

template <typename T, typename F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::enable_if_t<std::is_integral<std::decay_t<decltype(std::declval<T>().size())> >::value>
tagparfor_call_f (
#ifdef AMREX_USE_DPCPP
    sycl::nd_item<1> const& item,
#endif
    int i, T const& tag, F&& f) noexcept
{
    int N = tag.size();
#ifdef AMREX_USE_DPCPP
    f(item, i, N, tag);
#else
    f(      i, N, tag);
#endif
}


template <class TagType, class F>
void
ParallelFor_doit (Vector<TagType> const& tags, F && f)
{
    const int ntags = tags.size();
    if (ntags == 0) return;

    int ntotwarps = 0;
    Vector<int> nwarps;
    nwarps.reserve(ntags+1);
    for (int i = 0; i < ntags; ++i)
    {
        auto& tag = tags[i];
        nwarps.push_back(ntotwarps);
        ntotwarps += static_cast<int>((get_tag_size(tag)
                                       + Gpu::Device::warp_size-1) / Gpu::Device::warp_size);
    }
    nwarps.push_back(ntotwarps);

    std::size_t sizeof_tags = ntags*sizeof(TagType);
    std::size_t offset_nwarps = Arena::align(sizeof_tags);
    std::size_t sizeof_nwarps = (ntags+1)*sizeof(int);
    std::size_t total_buf_size = offset_nwarps + sizeof_nwarps;

    char* h_buffer = (char*)The_Pinned_Arena()->alloc(total_buf_size);
    char* d_buffer = (char*)The_Arena()->alloc(total_buf_size);

    std::memcpy(h_buffer, tags.data(), sizeof_tags);
    std::memcpy(h_buffer+offset_nwarps, nwarps.data(), sizeof_nwarps);
    Gpu::htod_memcpy_async(d_buffer, h_buffer, total_buf_size);

    auto d_tags = reinterpret_cast<TagType*>(d_buffer);
    auto d_nwarps = reinterpret_cast<int*>(d_buffer+offset_nwarps);

    constexpr int nthreads = 256;
    constexpr int nwarps_per_block = nthreads/Gpu::Device::warp_size;
    int nblocks = (ntotwarps + nwarps_per_block-1) / nwarps_per_block;

    amrex::launch(nblocks, nthreads, Gpu::gpuStream(),
#ifdef AMREX_USE_DPCPP
    [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
    AMREX_REQUIRE_SUBGROUP_SIZE(Gpu::Device::warp_size)
#else
    [=] AMREX_GPU_DEVICE () noexcept
#endif
    {
#ifdef AMREX_USE_DPCPP
        int g_tid = item.get_global_id(0);
#else
        int g_tid = blockDim.x*blockIdx.x + threadIdx.x;
#endif
        int g_wid = g_tid / Gpu::Device::warp_size;
        if (g_wid >= ntotwarps) return;

        int tag_id = amrex::bisect(d_nwarps, 0, ntags, g_wid);

        int b_wid = g_wid - d_nwarps[tag_id]; // b_wid'th warp on this box
#ifdef AMREX_USE_DPCPP
        int lane = item.get_local_id(0) % Gpu::Device::warp_size;
#else
        int lane = threadIdx.x % Gpu::Device::warp_size;
#endif
        int icell = b_wid*Gpu::Device::warp_size + lane;

#ifdef AMREX_USE_DPCPP
        tagparfor_call_f(item, icell, d_tags[tag_id], f);
#else
        tagparfor_call_f(      icell, d_tags[tag_id], f);
#endif
    });

    Gpu::streamSynchronize();
    The_Pinned_Arena()->free(h_buffer);
    The_Arena()->free(d_buffer);
}



template <class TagType, class F>
void
ParallelFor_doit2 (Vector<TagType> const& tags, F && f)
{
    const int ntags = tags.size();
    if (ntags == 0) return;

    int ntotwarps = 0;
    Vector<int> nwarps;
    nwarps.reserve(ntags+1);
    Vector<int> tagid(0);
    for (int i = 0; i < ntags; ++i)
    {
        auto& tag = tags[i];
        nwarps.push_back(ntotwarps);
        int Nwarps = static_cast<int>((get_tag_size(tag)
                                       + Gpu::Device::warp_size-1) / Gpu::Device::warp_size);
        ntotwarps += Nwarps;
        for (int j = 0; j < Nwarps; ++j) tagid.push_back(i);
    }
    nwarps.push_back(ntotwarps);

    std::size_t sizeof_tags = ntags*sizeof(TagType);
    std::size_t offset_nwarps = Arena::align(sizeof_tags);
    std::size_t sizeof_nwarps = (ntags+1)*sizeof(int);
    std::size_t offset_tagid = offset_nwarps + Arena::align(sizeof_nwarps);
    std::size_t sizeof_tagid = tagid.size()*sizeof(int);
    std::size_t total_buf_size = offset_tagid + sizeof_tagid;

    char* h_buffer = (char*)The_Pinned_Arena()->alloc(total_buf_size);
    char* d_buffer = (char*)The_Arena()->alloc(total_buf_size);

    std::memcpy(h_buffer, tags.data(), sizeof_tags);
    std::memcpy(h_buffer+offset_nwarps, nwarps.data(), sizeof_nwarps);
    std::memcpy(h_buffer+offset_tagid, tagid.data(), sizeof_tagid);
    Gpu::htod_memcpy_async(d_buffer, h_buffer, total_buf_size);

    auto d_tags = reinterpret_cast<TagType*>(d_buffer);
    auto d_nwarps = reinterpret_cast<int*>(d_buffer+offset_nwarps);
    auto d_tagid = reinterpret_cast<int*>(d_buffer+offset_tagid);

    constexpr int nthreads = 256;
    constexpr int nwarps_per_block = nthreads/Gpu::Device::warp_size;
    int nblocks = (ntotwarps + nwarps_per_block-1) / nwarps_per_block;

    amrex::launch(nblocks, nthreads, Gpu::gpuStream(),
#ifdef AMREX_USE_DPCPP
    [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
    AMREX_REQUIRE_SUBGROUP_SIZE(Gpu::Device::warp_size)
#else
    [=] AMREX_GPU_DEVICE () noexcept
#endif
    {
#ifdef AMREX_USE_DPCPP
        int g_tid = item.get_global_id(0);
#else
        int g_tid = blockDim.x*blockIdx.x + threadIdx.x;
#endif
        int g_wid = g_tid / Gpu::Device::warp_size;
        if (g_wid >= ntotwarps) return;

        //int tag_id = amrex::bisect(d_nwarps, 0, ntags, g_wid);
        int tag_id = d_tagid[g_wid];

        int b_wid = g_wid - d_nwarps[tag_id]; // b_wid'th warp on this box
#ifdef AMREX_USE_DPCPP
        int lane = item.get_local_id(0) % Gpu::Device::warp_size;
#else
        int lane = threadIdx.x % Gpu::Device::warp_size;
#endif
        int icell = b_wid*Gpu::Device::warp_size + lane;

#ifdef AMREX_USE_DPCPP
        tagparfor_call_f(item, icell, d_tags[tag_id], f);
#else
        tagparfor_call_f(      icell, d_tags[tag_id], f);
#endif
    });

    Gpu::streamSynchronize();
    The_Pinned_Arena()->free(h_buffer);
    The_Arena()->free(d_buffer);
}

template <class TagType, class StashStruct>
void
build_device_tags_nwarps (Vector<TagType> const& tags, StashStruct * stashed_buf)
{
    /* Build the device nwarps data structure */
    Vector<int> nwarps;
    nwarps.reserve(stashed_buf->ntags+1);
    stashed_buf->ntotwarps=0;
    for (int i = 0; i < stashed_buf->ntags; ++i)
    {
        auto& tag = tags[i];
        nwarps.push_back(stashed_buf->ntotwarps);
        stashed_buf->ntotwarps +=  static_cast<int>((detail::get_tag_size(tag)
                                                    + Gpu::Device::warp_size-1) / Gpu::Device::warp_size);
    }
    nwarps.push_back(stashed_buf->ntotwarps);

    std::size_t sizeof_tags = stashed_buf->ntags*sizeof(TagType);
    stashed_buf->offset_nwarps = Arena::align(sizeof_tags);
    stashed_buf->sizeof_nwarps = (stashed_buf->ntags+1)*sizeof(int);
    std::size_t total_buf_size = stashed_buf->offset_nwarps + stashed_buf->sizeof_nwarps;

    /* Allocate pinned/device memory for the tags and the device nwarps data strcture */
    stashed_buf->d_buffer = (char*)The_Arena()->alloc(total_buf_size);
    stashed_buf->h_buffer = (char*)The_Pinned_Arena()->alloc(total_buf_size);

    /* transfer to device */
    std::memcpy(stashed_buf->h_buffer, tags.data(), sizeof_tags);
    std::memcpy(stashed_buf->h_buffer + stashed_buf->offset_nwarps, nwarps.data(), stashed_buf->sizeof_nwarps);
    Gpu::htod_memcpy_async(stashed_buf->d_buffer, stashed_buf->h_buffer, total_buf_size);

}

template <class TagType, class F>
void
ParallelFor_doit (Vector<TagType> const& tags, char * d_buffer, int ntotwarps,
                  int ntags, std::size_t offset_nwarps, F && f)
{
    constexpr int nthreads = 256;
    constexpr int nwarps_per_block = nthreads/Gpu::Device::warp_size;
    int nblocks = (ntotwarps + nwarps_per_block-1) / nwarps_per_block;

    auto d_tags = reinterpret_cast<TagType*>(d_buffer);
    auto d_nwarps = reinterpret_cast<int*>(d_buffer + offset_nwarps);

    amrex::launch(nblocks, nthreads, Gpu::gpuStream(),
#ifdef AMREX_USE_DPCPP
    [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
    AMREX_REQUIRE_SUBGROUP_SIZE(Gpu::Device::warp_size)
#else
    [=] AMREX_GPU_DEVICE () noexcept
#endif
    {
#ifdef AMREX_USE_DPCPP
        int g_tid = item.get_global_id(0);
#else
        int g_tid = blockDim.x*blockIdx.x + threadIdx.x;
#endif
        int g_wid = g_tid / Gpu::Device::warp_size;
        if (g_wid >= ntotwarps) return;

        int tag_id = amrex::bisect(d_nwarps, 0, ntags, g_wid);

        int b_wid = g_wid - d_nwarps[tag_id]; // b_wid'th warp on this box
#ifdef AMREX_USE_DPCPP
        int lane = item.get_local_id(0) % Gpu::Device::warp_size;
#else
        int lane = threadIdx.x % Gpu::Device::warp_size;
#endif
        int icell = b_wid*Gpu::Device::warp_size + lane;

        TagType tag = d_tags[tag_id];
        if (icell>=tag.box().numPts()) return;

#ifdef AMREX_USE_DPCPP
        tagparfor_call_f(item, icell, tag, f);
#else
        tagparfor_call_f(      icell, tag, f);
#endif
    });

    Gpu::streamSynchronize();
}

}

template <class TagType, class F>
std::enable_if_t<std::is_same<std::decay_t<decltype(std::declval<TagType>().box())>,
                              Box>::value>
ParallelFor (Vector<TagType> const& tags, int ncomp, F && f)
{
    detail::ParallelFor_doit(tags,
        [=] AMREX_GPU_DEVICE (
#ifdef AMREX_USE_DPCPP
            sycl::nd_item<1> const& /*item*/,
#endif
            int icell, int ncells, int i, int j, int k, TagType const& tag) noexcept
        {
            if (icell < ncells) {
                for (int n = 0; n < ncomp; ++n) {
                    f(i,j,k,n,tag);
                }
            }
        });
}

template <class TagType, class F>
std::enable_if_t<std::is_same<std::decay_t<decltype(std::declval<TagType>().box())>, Box>::value>
ParallelFor (Vector<TagType> const& tags, F && f)
{
    detail::ParallelFor_doit(tags,
        [=] AMREX_GPU_DEVICE (
#ifdef AMREX_USE_DPCPP
            sycl::nd_item<1> const& /*item*/,
#endif
            int icell, int ncells, int i, int j, int k, TagType const& tag) noexcept
        {
            if (icell < ncells) {
                f(i,j,k,tag);
            }
        });
}

template <class TagType, class F>
std::enable_if_t<std::is_integral<std::decay_t<decltype(std::declval<TagType>().size())> >::value>
ParallelFor (Vector<TagType> const& tags, F && f)
{
    detail::ParallelFor_doit(tags,
        [=] AMREX_GPU_DEVICE (
#ifdef AMREX_USE_DPCPP
            sycl::nd_item<1> const& /*item*/,
#endif
            int icell, int ncells, TagType const& tag) noexcept
        {
            if (icell < ncells) {
                f(icell,tag);
            }
        });
}

#endif

}

#endif
