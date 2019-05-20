import pyopencl as cl
import numpy as np
import json

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def loadProgram(ctx, file):
    f = open(file, "r")
    ll = "\n".join(f.readlines())
    f.close()
    prog = cl.Program(ctx, ll)

    try:
        prog.build()
    except:
        print("Error: ")
        print(prog.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
        raise
    
    return prog

#def saveImage(np_buffer):
#    ii.imwrite("test.jpg", np_buffer)

prog = loadProgram(ctx, "projekt_opencl.cl")

example_input = [ [10.0, 14.0, 18.0],[7.0, 6.5, 12.0],[3.0, 1.0, 0.4] ]
example_input_size = [3,3]
example_output_size = [64,64]

np_input = np.array(example_input, dtype=np.float32)
np_input_size = np.array(example_input_size, dtype=np.uint32)
np_output_size = np.array(example_output_size, dtype=np.uint32)

input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, np_input.nbytes, hostbuf = np_input)
input_size = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, np_input_size.nbytes, hostbuf = np_input_size)
output_size = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, np_output_size.nbytes, hostbuf = np_output_size)
output_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, example_output_size[0] * example_output_size[1] * 4)

prog.idw2(queue, (24,24), None, input, input_size, output_buffer, output_size)

output = np.zeros( (example_output_size[0], example_output_size[1]), dtype = np.float32 )
cl.enqueue_copy(queue, output, output_buffer).wait()

#saveImage(output)
with open('ocldump.json', 'w') as f:
    json.dump(output, f)