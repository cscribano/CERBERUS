import ctypes
import cupy as cp
import cupyx
import tensorrt as trt
import logging

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
LOGGER = logging.getLogger(__name__)

class HostDeviceMem:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self.host = cupyx.empty_pinned(size, dtype)
        self.device = cp.empty(size, dtype)

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    @property
    def nbytes(self):
        return self.host.nbytes

    @property
    def hostptr(self):
        return self.host.ctypes.data

    @property
    def devptr(self):
        return self.device.data.ptr

    def copy_htod_async(self, stream):
        self.device.data.copy_from_host_async(self.hostptr, self.nbytes, stream)

    def copy_dtoh_async(self, stream):
        self.device.data.copy_to_host_async(self.hostptr, self.nbytes, stream)

class TRTModel:
    """Base class for TRT models.

    Attributes
    ----------
    PLUGIN_PATH : Path, optional
        Path to TensorRT plugin.
    ENGINE_PATH : Path
        Path to TensorRT engine.
        If not found, TensorRT engine will be converted from the ONNX model
        at runtime and cached for later use.
    MODEL_PATH : Path
        Path to ONNX model.
    INPUT_SHAPE : tuple
        Input size in the format `(channel, height, width)`.
    OUTPUT_LAYOUT : int
        Feature dimension output by the model.
    """
    __registry = {}

    PLUGIN_PATH = None
    ENGINE_PATH = None
    MODEL_PATH = None
    INPUT_SHAPE = None
    OUTPUT_LAYOUT = None

    def __init_subclass__(cls, model=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__registry[cls.__name__] = cls

    @classmethod
    def get_model(cls, name):
        return cls.__registry[name]

    @classmethod
    def build_engine(cls, trt_logger, batch_size):
        with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, trt_logger) as parser:

            builder.max_batch_size = batch_size
            LOGGER.info('Building engine with batch size: %d', batch_size)
            LOGGER.info('This may take a while...')

            # parse model file
            with open(cls.MODEL_PATH, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    LOGGER.critical('Failed to parse the ONNX file')
                    for err in range(parser.num_errors):
                        LOGGER.error(parser.get_error(err))
                    return None

            # reshape input to the right batch size
            net_input = network.get_input(0)
            assert cls.INPUT_SHAPE == net_input.shape[1:]
            net_input.shape = (batch_size, *cls.INPUT_SHAPE)

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            if builder.platform_has_fast_fp16:
                LOGGER.debug("TensorRT is using FP16")
                config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()
            profile.set_shape(
                net_input.name,                  # input tensor name
                (batch_size, *cls.INPUT_SHAPE),  # min shape
                (batch_size, *cls.INPUT_SHAPE),  # opt shape
                (batch_size, *cls.INPUT_SHAPE)   # max shape
            )
            config.add_optimization_profile(profile)

            # engine = builder.build_cuda_engine(network)
            engine = builder.build_engine(network, config)
            if engine is None:
                LOGGER.critical('Failed to build engine')
                return None

            LOGGER.info("Completed creating engine")
            with open(cls.ENGINE_PATH, 'wb') as engine_file:
                engine_file.write(engine.serialize())
            return engine

class TRTInference:
    # initialize TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

        # load plugin if the model requires one
        if self.model.PLUGIN_PATH is not None:
            try:
                ctypes.cdll.LoadLibrary(self.model.PLUGIN_PATH)
            except OSError as err:
                raise RuntimeError('Plugin not found') from err

        # load trt engine or build one if not found
        if not self.model.ENGINE_PATH.exists():
            print("Building TRT Model...")
            self.engine = self.model.build_engine(TRTInference.TRT_LOGGER, self.batch_size)
        else:
            runtime = trt.Runtime(TRTInference.TRT_LOGGER)
            with open(self.model.ENGINE_PATH, 'rb') as engine_file:
                self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError('Unable to load the engine file')
        if self.engine.has_implicit_batch_dimension:
            assert self.batch_size <= self.engine.max_batch_size
        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream()

        # allocate buffers
        self.bindings = []
        self.outputs = []
        self.input = None
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # allocate host and device buffers
            buffer = HostDeviceMem(size, dtype)
            # append the device buffer to device bindings
            self.bindings.append(buffer.devptr)
            if self.engine.binding_is_input(binding):
                if not self.engine.has_implicit_batch_dimension:
                    assert self.batch_size == shape[0]
                # expect one input
                self.input = buffer
            else:
                self.outputs.append(buffer)
        assert self.input is not None

        # timing events
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()

        print("TensorRT Model Ready!")

    def __del__(self):
        if hasattr(self, 'context'):
            self.context.__del__()
        if hasattr(self, 'engine'):
            self.engine.__del__()

    def infer(self):
        self.infer_async()
        return self.synchronize()

    def infer_async(self, from_device=False):
        self.start.record(self.stream)
        if not from_device:
            self.input.copy_htod_async(self.stream)
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings,
                                       stream_handle=self.stream.ptr)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.ptr)
        for out in self.outputs:
            out.copy_dtoh_async(self.stream)
        self.end.record(self.stream)

    def synchronize(self):
        self.stream.synchronize()
        return [out.host for out in self.outputs]

    def get_infer_time(self):
        self.end.synchronize()
        return cp.cuda.get_elapsed_time(self.start, self.end)

