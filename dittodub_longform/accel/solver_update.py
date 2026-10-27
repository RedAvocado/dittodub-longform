"""Screen exact solver-update fusion and once-per-latent block validation."""
import mlx.core as mx

UPDATE=mx.fast.metal_kernel(name='ming_solver_update_v7',
    input_names=['y','prediction','noise','dt','scale','cfg'],output_names=['out'],
    header='#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n',
    compile_options={'math_mode':'safe'},source='''
    uint i=thread_position_in_grid.x;
    float pred=prediction[i],null_pred=prediction[64+i];
    float delta=pred-null_pred;
    float scaled=delta*cfg;
    float derivative=pred+scaled;
    float dy=dt*derivative;
    float next=y[i]+dy;
    float shift=scale*noise[i];
    out[i]=next+shift;
    ''')


def update(y,prediction,noise,dt,scale,cfg):
    return UPDATE(inputs=[y,prediction,noise,dt,scale,mx.array(cfg,dtype=mx.float32)],
        grid=(64,1,1),threadgroup=(64,1,1),output_shapes=[y.shape],output_dtypes=[mx.float32])[0]


def first_blocks(full,dit,compiled_blocks):
    for block in dit.blocks:
        supported=block.supported(full,None,dit.rotary_embed)
        full=block(full,None,dit.rotary_embed)
        compiled_blocks.append(block.__dict__['_compiled_v6'][1] if supported
            else lambda x,block=block:block(x,None,dit.rotary_embed))
    return full

