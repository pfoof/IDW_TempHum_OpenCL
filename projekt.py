import sys
import pyopencl as cl
import numpy as np
import json
from scipy.misc import imsave

# Settings

DUMP_JSON = False
MAKE_IMAGE = True
PROFILING = True
OUTPUT_SIZE = [ 1024, 1024 ]
IDW_GWS = (128, 128)
IMAGE_GWS = (128, 128)
LOAD_FILE = 'input.json'

if len(sys.argv) > 1:
    gws = int(sys.argv[1])
    IDW_GWS = (gws, gws)
    IMAGE_GWS = (gws, gws)

# endof Settings

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties = cl.command_queue_properties.PROFILING_ENABLE) if PROFILING else cl.CommandQueue(ctx)
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

def saveImage(np_buffer, index):
    imsave("test_%d.png" % index, np_buffer)

def loadData(file):
    with open(file, 'r') as f:
        arr = json.load(f)
        return (len(arr), arr)

# Main

prog = loadProgram(ctx, "projekt_opencl.cl")

arrlen, arr = loadData(LOAD_FILE)
example_output_size = OUTPUT_SIZE

for (i, record) in enumerate(arr):
    example_input_size, example_input = record['size'], record['data']
    
    np_input = np.array(example_input, dtype=np.float32)
    np_input_size = np.array(example_input_size, dtype=np.uint32)
    np_output_size = np.array(example_output_size, dtype=np.uint32)
    np_image = np.zeros( (example_output_size[1], example_output_size[0], 4), dtype=np.uint8 )

    input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, np_input.nbytes, hostbuf = np_input)
    input_size = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, np_input_size.nbytes, hostbuf = np_input_size)
    output_size = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, np_output_size.nbytes, hostbuf = np_output_size)
    output_buffer = cl.Buffer(ctx, mf.READ_WRITE, example_output_size[0] * example_output_size[1] * 4)
    if MAKE_IMAGE:
        image_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, np_image.nbytes)

    idw2event = prog.idw2(queue, IDW_GWS, None, input, input_size, output_buffer, output_size)
    idw2event.wait()
    if MAKE_IMAGE:
        colorizeevent = prog.colorize(queue, IMAGE_GWS, None, output_buffer, image_buffer, output_size)
        colorizeevent.wait()

    output = np.zeros( (example_output_size[1], example_output_size[0]), dtype = np.float32 )
    outputenqueue = cl.enqueue_copy(queue, output, output_buffer)
    outputenqueue.wait()
    if MAKE_IMAGE:
        imageenqueue = cl.enqueue_copy(queue, np_image, image_buffer)
        imageenqueue.wait()

        saveImage(np_image, i)

    if DUMP_JSON:
        with open("ocldump_%d.json" % i, 'w') as f:
            json.dump(output.tolist(), f, indent=4)

    if PROFILING:
        with open("profiling.txt", 'a') as f:
            idw2start = idw2event.get_profiling_info(cl.profiling_info.START)
            idw2end = idw2event.get_profiling_info(cl.profiling_info.END)
            print("idw2: %.3f" % ((idw2end - idw2start) / 1000.0) )
            colorizestart = colorizeevent.get_profiling_info(cl.profiling_info.START)
            colorizeend = colorizeevent.get_profiling_info(cl.profiling_info.END)
            print("colorize: %.3f" % ((colorizeend - colorizestart) / 1000.0) )
            outputstart = outputenqueue.get_profiling_info(cl.profiling_info.START)
            outputend = outputenqueue.get_profiling_info(cl.profiling_info.END)
            print("output_copy: %.3f" % ((outputend - outputstart) / 1000.0) )
            imagestart = imageenqueue.get_profiling_info(cl.profiling_info.START)
            imageend = imageenqueue.get_profiling_info(cl.profiling_info.END)
            print("image_copy: %.3f" % ((imageend - imagestart) / 1000.0) )
            f.write(" \n---------------\n[ %d x %d ]\t\t->[ %d x %d ] \n" % ( example_input_size[0], example_input_size[1], example_output_size[0], example_output_size[1] ) )
            f.write(" GWS: idw = %d, %d\timage = %d, %d\n" % (IDW_GWS + IMAGE_GWS) )
            
            LMS = ctx.devices[0].get_info(cl.device_info.LOCAL_MEM_SIZE) // 1024
            GMS = ctx.devices[0].get_info(cl.device_info.GLOBAL_MEM_SIZE) // (1024 * 1024)
            EU = ctx.devices[0].get_info(cl.device_info.MAX_COMPUTE_UNITS)
            FREQ = ctx.devices[0].get_info(cl.device_info.MAX_CLOCK_FREQUENCY)
            DEV_NAME = ctx.devices[0].get_info(cl.device_info.NAME)
            f.write(" Device: %s \n" % DEV_NAME )
            f.write(" MEM: local = %d kB, global = %d MB, EU = %d @%dMHz\n" % (LMS, GMS, EU, FREQ) )
            
            f.write(" \nIDW2\t\t\t%.3f\n" % ((idw2end - idw2start) / 1000.0) )
            f.write(" Colorize\t\t%.3f\n" % ((colorizeend - colorizestart) / 1000.0) )
            f.write(" Output_copy\t%.3f\n" % ((outputend - outputstart) / 1000.0) )
            f.write(" Image_copy\t\t%.3f\n" % ((imageend - imagestart) / 1000.0) )
