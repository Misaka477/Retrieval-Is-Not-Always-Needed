"""Fix B_BWD and f_bwd for single buffer output."""
import re

with open('rina/kernels/light.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_bwd_decl = '''    B_BWD = ("void k3_light_bwd(float*,float*,float*,float*,float*,float*,float*,"
             "const float*,const float*,const float*,const float*,"
             "const float*,const float*,const float*,const float*,"
             "const float*,const float*,const float*,const float*,const float*,const float*,const float*,"
             "int,int,int);"
             "void f_bwd(torch::Tensor g_hc,torch::Tensor g_fo,torch::Tensor g_ls,"
             "torch::Tensor g_nw,torch::Tensor g_nb,torch::Tensor g_Po,torch::Tensor g_ft,"
             "torch::Tensor g_ho,torch::Tensor g_hf,torch::Tensor h,torch::Tensor x,"
             "torch::Tensor h_fast,torch::Tensor fm,torch::Tensor nm,torch::Tensor a_g,"
             "torch::Tensor P,torch::Tensor fmw,torch::Tensor fmb,"
             "torch::Tensor nw,torch::Tensor nb,torch::Tensor sw,torch::Tensor sb){"
             "k3_light_bwd(g_hc.data_ptr<float>(),g_fo.data_ptr<float>(),g_ls.data_ptr<float>(),"
             "g_nw.data_ptr<float>(),g_nb.data_ptr<float>(),g_Po.data_ptr<float>(),g_ft.data_ptr<float>(),"
             "g_ho.data_ptr<float>(),g_hf.data_ptr<float>(),h.data_ptr<float>(),x.data_ptr<float>(),"
             "h_fast.data_ptr<float>(),fm.data_ptr<float>(),nm.data_ptr<float>(),a_g.data_ptr<float>(),"
             "P.data_ptr<float>(),fmw.data_ptr<float>(),fmb.data_ptr<float>(),"
             "nw.data_ptr<float>(),nb.data_ptr<float>(),sw.data_ptr<float>(),sb.data_ptr<float>(),"
             "h.size(0),h.size(1),P.size(0));}")'''

new_bwd_decl = '''    B_BWD = ("void k3_light_bwd(float*,"
             "const float*,const float*,const float*,const float*,"
             "const float*,const float*,const float*,const float*,"
             "const float*,const float*,const float*,const float*,const float*,const float*,const float*,"
             "int,int,int);"
             "void f_bwd(torch::Tensor grad_out,"
             "torch::Tensor g_ho,torch::Tensor g_hf,torch::Tensor h,torch::Tensor x,"
             "torch::Tensor h_fast,torch::Tensor fm,torch::Tensor nm,torch::Tensor a_g,"
             "torch::Tensor P,torch::Tensor fmw,torch::Tensor fmb,"
             "torch::Tensor nw,torch::Tensor nb,torch::Tensor sw,torch::Tensor sb){"
             "k3_light_bwd(grad_out.data_ptr<float>(),"
             "g_ho.data_ptr<float>(),g_hf.data_ptr<float>(),h.data_ptr<float>(),x.data_ptr<float>(),"
             "h_fast.data_ptr<float>(),fm.data_ptr<float>(),nm.data_ptr<float>(),a_g.data_ptr<float>(),"
             "P.data_ptr<float>(),fmw.data_ptr<float>(),fmb.data_ptr<float>(),"
             "nw.data_ptr<float>(),nb.data_ptr<float>(),sw.data_ptr<float>(),sb.data_ptr<float>(),"
             "h.size(0),h.size(1),P.size(0));}")'''

if old_bwd_decl in content:
    content = content.replace(old_bwd_decl, new_bwd_decl)
    with open('rina/kernels/light.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Updated successfully')
else:
    print('Could not find old B_BWD declaration')
    # Show what the B_BWD looks like
    idx = content.find('B_BWD')
    if idx > 0:
        print('Found B_BWD at', idx)
        print(content[idx:idx+800])
