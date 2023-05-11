import fastarch.dataflow_convolution as dc
import fastarch.build_hardware_v2 as bh
import fastarch.build_models_v2 as bm
import fastarch.conv_helper as ch
import fastarch.dataflow_wrapper as dw
import fastarch.evolutionary_search_v3 as es
import fastarch.dataflow_estimator_conv as dec
import fastarch.accel_only_evolutionary_search as aoes

aoes.test()
#dec.test()
#model = bm.get_DeiT_Tiny(1, 1.0, 1.0, 0.0)
#es.search_model(model, 50, 1)

#model = bm.get_DeiT_Tiny(1, 1.0, 1.0, 0.0)
#layer_set = bm.model_to_layer_set(model)
#layer_set.print()

#dw.run_levit_128(pipelining=False, bw=77)
#dw.run_deit_tiny(pipelining=False, bw=77)
#input()
#dw.run_deit_tiny(pipelining=False, bw=10)
#input()
#dw.run_deit_tiny(pipelining=False, bw=5)
#input()

hw_config = bh.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=77, on_chip_bandwidth=100, total_sram_size=1000000)
layer = ch.ConvLayer(rows=224, cols=224, c_in=1, c_out=8, filter_dim=16, step_size=16)
#print(layer.prows, layer.pcols)
# [rows/cols, x/y/ci/co, x/y/ci/co, x/y/ci/co, x/y/ci/co, t_x, t_y, t_ci, t_co, c_x, c_y, c_ci, c_kx, c_ky, c_co]
params = ["rows", "co", "ci", "y", "x", 120, 120, 1, 1, "x", "y", "ci", "kx", "ky", "co", 7, 1, 3, 4, 1, 1]

mm_layer = bm.Layer(A_rows = layer.out_rows * layer.out_cols, A_cols_B_rows = layer.filter_dim ** 2 * layer.c_in, B_cols = layer.c_out)
mm_params = ['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10]

#conv_res = dc.run_conv_dataflow(hw_config, layer, params)
#mm_res = dw.run_layer(hw_config, mm_params, mm_layer)
#print(conv_res)
#print(mm_res)
#print(layer.get_flops())
#print(mm_layer.get_flops())