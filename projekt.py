import pyopencl as cl
import numpy as np
import json
from scipy.misc import imsave

# Settings

DUMP_JSON = False
MAKE_IMAGE = True
OUTPUT_SIZE = [ 256, 256 ]
IDW_GWS = (24, 24)
IMAGE_GWS = (24, 24)
LOAD_FILE = 'input.json'

# endof Settings

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# Helper functions

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

def saveImage(np_buffer):
    imsave("test.jpg", np_buffer)

def loadData(file):
    with open(file, 'r') as f:
        d = json.load(f)
        return (d['size'], d['data'])

# Main

prog = loadProgram(ctx, "projekt_opencl.cl")

example_input_size, example_input = loadData(LOAD_FILE)
example_output_size = OUTPUT_SIZE

np_input = np.array(example_input, dtype=np.float32)
np_input_size = np.array(example_input_size, dtype=np.uint32)
np_output_size = np.array(example_output_size, dtype=np.uint32)
np_image = np.zeros( (example_output_size[0], example_output_size[1], 3), dtype=np.uint8 )

input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, np_input.nbytes, hostbuf = np_input)
input_size = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, np_input_size.nbytes, hostbuf = np_input_size)
output_size = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, np_output_size.nbytes, hostbuf = np_output_size)
output_buffer = cl.Buffer(ctx, mf.READ_WRITE, example_output_size[0] * example_output_size[1] * 4)
if MAKE_IMAGE:
    image_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, np_image.nbytes)

prog.idw2(queue, IDW_GWS, None, input, input_size, output_buffer, output_size).wait()
if MAKE_IMAGE:
    prog.colorize(queue, IMAGE_GWS, None, output_buffer, image_buffer, output_size).wait()

output = np.zeros( (example_output_size[0], example_output_size[1]), dtype = np.float32 )
cl.enqueue_copy(queue, output, output_buffer).wait()
if MAKE_IMAGE:
    cl.enqueue_copy(queue, np_image, image_buffer).wait()

    saveImage(np_image)

if DUMP_JSON:
    with open('ocldump.json', 'w') as f:
        json.dump(output.tolist(), f, indent=4)