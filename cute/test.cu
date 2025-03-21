
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cute/stride.hpp"
#include "cute/underscore.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include <vector>


using namespace cute;


template<int N>
using CLayout_64xN   = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,Int<N/8>>>,
                            Stride<Stride<_128,_1,_16>,Stride<_64,_8,   _512>>>;

using CLayout_64x128 = CLayout_64xN<128>;

template <int M, int K>
using ABLayout       = Layout<Shape <_128,Shape <Int<M>,Int<K>>>,
                            Stride<  _0,Stride<    _1,Int<M>>>>;


int test_mma(){
    std::cout << "test_mma..." << std::endl;
    using ALayout = ABLayout<64, 32>;
    using BLayout = ABLayout<128, 32>;
    using CLayout = CLayout_64x128;
    std::cout << "ALayout = " << ALayout{} << std::endl;
    std::cout << "BLayout = " << BLayout{} << std::endl;
    std::cout << "CLayout = " << CLayout{} << std::endl;

    return 0;
}

int test_tma(){
    std::cout << "test_tma..." << std::endl;
    using StrideA = cute::Stride<_1024, _1, _1>;
    auto repeat_shape = repeat_like(StrideA{}, int32_t(0));
    std::cout << "repeat_shape = " << repeat_shape << std::endl;

    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _2, _1>;
    using GmemTiledCopyA = SM90_TMA_LOAD;
    using GmemTiledCopyB = SM90_TMA_LOAD_MULTICAST;
    // using SmemLayoutA = 
    // using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
    //   GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    
    // using TMA_A = decltype(make_tma_copy_A_sm90(
    //     GmemTiledCopyA{},
    //     make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
    //     SmemLayoutA{}(_,_,0),
    //     TileShape{},
    //     ClusterShape{}));
    auto gtensor = make_tensor(static_cast<cutlass::float_e4m3_t const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{});
    auto cta_tiler_mk = remove<1>(TileShape{});
    std::cout << "shape(gtensor) = " << shape(gtensor) << std::endl;
    std::cout << "stride(gtensor) = " << stride(gtensor) << std::endl;
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler_mk);
    // auto cta_t_tile = make_layout(cluster_size_n);
    std::cout << "cta_v_tile = " << cta_v_tile << std::endl;

    // new: create a TMA load multicast object for the given cluster size
    // create the GMEM tensor
    using gmem_layout = Layout<Shape<_512, _128>, Stride<_128, _1>>;
    auto gmem_tensor = make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), gmem_layout{});
    // create the SMEM layout
    using smem_layout = Layout<Shape<_16, _16>, Stride<_16, _1>>;
  
    // auto tma_load = make_tma_copy(SM90_TMA_LOAD_MULTICAST{},
    //     gmem_tensor, smem_layout{}, cute::_2{});
    
    // std::cout << "tma_load = " << tma_load << std::endl;

    return 0;
}

int tests_scale(){
    std::cout << "tests_scale..." << std::endl;
    
    auto M = Int<500>{};
    auto N = Int<511>{};
    auto K = Int<500>{};
    auto ScaleGranularityM = Int<1>{};
    auto tK = Int<64>{};
    auto L = Int<1>{};
    auto tN = Int<64>{};
    auto scaleA_shape = make_shape(M / ScaleGranularityM, tK, L); // (scale_m,k,l)
    auto scaleA_layout = make_ordered_layout(scaleA_shape,  Step<_0, _1, _2>{});
    auto scaleB_shape = make_shape(tN, tK, L); // (n,k,l)
    auto scaleB_layout = make_ordered_layout(scaleB_shape, Step<_1, _0, _2>{});
    std::cout << "scaleA_layout = " << scaleA_layout << std::endl;
    std::cout << "scaleB_layout = " << scaleB_layout << std::endl;
    std::vector<int8_t> scaleA_data(128*128*1);
    Tensor mScaleA_mkl = make_tensor(scaleA_data.data(), scaleA_layout); // (scale_m,k,l)
    Tensor mScaleB_nkl = make_tensor(scaleA_data.data(), scaleB_layout); // (n,k,l)
    static constexpr int ScaleMsPerTile = 128;
    Tensor gScaleA = local_tile( 
      mScaleA_mkl, make_tile(Int<ScaleMsPerTile>{}), 
      make_coord(1,_,1));                   // (ScaleMsPerTile,k,1)
    // Tensor cScaleA = local_tile( 
    //   cScaleA_mkl, make_tile(Int<ScaleMsPerTile>{}), 
    //   make_coord(m_coord,_,l_coord));
    // Tensor gScaleB = mScaleB_nkl(n_coord,_,l_coord);   
    std::cout << "gScaleA = " << gScaleA.layout() << std::endl;

    auto mn_shape = make_shape(M, K, L); // (scale_m,k,l)
    auto mn_layout = make_ordered_layout(mn_shape,  Step<_0, _1, _2>{});
    Tensor mn_mkl = make_tensor(scaleA_data.data(), mn_layout);
    using tile_shape = Shape<_128, _128, _128>;
    using X = Underscore;
    Tensor mn_tensor = local_tile( 
      mn_mkl, tile_shape{}, 
      make_coord(_,_,_), Step<_1, X,_1>{});
    std::cout << "mn_tensor = " << mn_tensor.layout() << std::endl;
    return 0;
}

int test_tiled_product(){
    std::cout << "test_tiled_product..." << std::endl;
    using AtomThrID = Layout<_128>;
    using AtomLayoutMNK = Layout<Shape<_2,_1,_1>>;
    using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
    std::cout << "AtomThrID = " << AtomThrID{} << std::endl;
    std::cout << "AtomLayoutMNK = " << AtomLayoutMNK{} << std::endl;
    std::cout << "ThrLayoutVMNK = " << ThrLayoutVMNK{} << std::endl;
    return 0;
}

int test_copy(){
    // Block scaling gmem-to-smem copy atom 
  using SmemBlockScalingCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<float>, float>;
  using SmemBlockScalingCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<float>, float>;
  TiledCopy scale_copy_a = make_tiled_copy(SmemBlockScalingCopyAtomA{}, 
      Layout<Shape<_32, _1>>{}, Layout<Shape<_4, _1>>{}); // (1,1,1)
  TiledCopy scale_copy_b = make_tiled_copy(SmemBlockScalingCopyAtomB{}, 
      Layout<Shape<_1>>{}, Layout<Shape<_1>>{}); // (1,1,1)
//   ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(threadIdx.x);
//   ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(threadIdx.x);
}
int main(){
    std::vector<float> vec{1,2,3,4,5,6,7,8,9,10, 11, 12};
    int M = 3;
    int K = 4;
    int L = 1;
    auto shape = cute::make_shape(M, K, L);
    auto stride = cute::make_stride(K, _1{}, 0);
    auto layout = cute::make_layout(shape, stride);
    cute::Tensor tensor = cute::make_tensor(vec.data(), layout);
    std::cout << "tensor = " << tensor << std::endl;
    using TileShape = Shape<_2, _2, _2>;
    using ClusterShape = Shape<_1, _2, _1>;
    // using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
    // using SmemLayoutA = decltype(tile_to_shape(
    //   SmemLayoutAtomA{},
    //   make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
    //   cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
    // using TMA_A = decltype(make_tma_copy_A_sm90(
    //     GmemTiledCopyA{},
    //     make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
    //     SmemLayoutA{}(_,_,0),
    //     TileShape{},
    //     ClusterShape{}));

    using X = Underscore;
    auto slice_coord = make_coord(_,_,_);
    using step = Step<_1, X,_1>;
    std::cout << "step = " << step{} << std::endl;
    Tensor gA_mkl = local_tile(tensor, TileShape{}, slice_coord, step{});
    std::cout << "gA_mkl = " << gA_mkl << std::endl;

    auto scaleA_shape = make_shape(M / 1, K, L); // (scale_m,k,l)
    auto scaleA_layout = make_ordered_layout(scaleA_shape,  Step<_0, _1, _2>{});
    std::cout << "scaleA_layout = " << scaleA_layout << std::endl;

    auto scaleB_shape = make_shape(12, K, L); // (n,k,l)
    auto scaleB_layout = make_ordered_layout(scaleB_shape, Step<_1, _0, _2>{});
    std::cout << "scaleB_layout = " << scaleB_layout << std::endl;

    test_mma();
    test_tma();
    tests_scale();
    return 0;
}