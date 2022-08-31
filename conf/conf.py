# -*- coding: utf-8 -*-
# ---------------------

import os

PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
	os.environ['PYTHONPATH'] = PYTHONPATH
else:
	os.environ['PYTHONPATH'] += (':' + PYTHONPATH)

import json
import datetime
import pytz
import string

import socket
import random
import torch
import numpy as np
from path import Path
from typing import Optional
from types import SimpleNamespace
from collections.abc import Mapping

def set_seed(seed=None):
	# type: (Optional[int]) -> int
	"""
	set the random seed using the required value (`seed`)
	or a random value if `seed` is `None`
	:return: the newly set seed
	"""
	if seed is None:
		seed = random.randint(1, 10000)

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	return seed

def get_unique_identifier(length: int = 8) -> str:
	"""Create a unique identifier by choosing `length`
	random characters from list of ascii characters and numbers
	"""
	alphabet = string.ascii_lowercase + string.digits
	uuid = "".join(alphabet[ix] for ix in np.random.choice(len(alphabet), length))
	return uuid

def find_free_port():
	s = socket.socket()
	s.bind(('', 0))            # Bind to a free port provided by the host.
	return s.getsockname()[1]  # Return the port number assigned.

class ConfigDecoder(json.JSONDecoder):
	def __init__(self, **kwargs):
		json.JSONDecoder.__init__(self, **kwargs)
		# Use the custom JSONArray
		self.parse_array = self.JSONArray
		# Use the python implemenation of the scanner
		self.scan_once = json.scanner.py_make_scanner(self)

	def JSONArray(self, s_and_end, scan_once, **kwargs):
		values, end = json.decoder.JSONArray(s_and_end, scan_once, **kwargs)
		return tuple(values), end

class DefaultNamespace(SimpleNamespace, Mapping):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def __iter__(self):
		return self.__dict__.__iter__()

	def __len__(self):
		return self.__dict__.__len__()

	def __getitem__(self, item):
		return self.__dict__.__getitem__(item)

	def json(self):
		o = json.dumps(self, default=lambda o: getattr(o, '__dict__', str(o)))
		return o

	def todict(self):
		o = json.loads(self.json())
		return o

	def get(self, key, default=None):
		try:
			y = self.__dict__[key]
			return y
		except KeyError:
			return default


class Conf(object):
	# HOSTNAME = socket.gethostname()
	HOSTNAME = socket.gethostname()  # socket.gethostbyname(socket.gethostname())
	PORT = find_free_port()
	OUT_PATH = Path(__file__).parent.parent

	def __init__(self, conf_file_path=None, data_root=None, seed=None, exp_name=None,
				 resume=False, log_each_step=True, log=True, device='cuda'):
		# type: (str, int, str, str, bool, bool, bool, str) -> None
		"""
		:param conf_file_path: optional path of the configuration file, if `resume` is true then this
			will be the path to the experiment log dir
		:param data_root: Overrides data_root from configuration file
		:param seed: desired seed for the RNG; if `None`, it will be chosen randomly
		:param exp_name: name of the experiment
		:param resume: `False` to start a new experiment, `True` to resume an existing one
		:param log: `False` if you want to enable experiment logging; `False` otherwise
		:param log_each_step: `True` if you want to log each step; `False` otherwise
		:param device: torch device you want to use for train/test
			:example values: 'cpu', 'cuda', 'cuda:5', ...
		"""
		self.exp_name = exp_name
		self.log_enabled = log
		self.log_each_step = log_each_step
		self.device = device
		self.resume = resume

		self.hostname = Conf.HOSTNAME
		self.port = Conf.PORT

		# Placeholders, warning: MUST call setup_device_id before any training can happen!
		self.rank = 0
		self.local_rank = -1

		# Check if we are running a slurm job
		self.slurm = os.environ.get("SLURM_TASK_PID") is not None
		if self.slurm:
			print(">> Detected SLURM")
			self.tmpdir = os.environ.get("TMPDIR")
		else:
			self.tmpdir = None

		# DDP STUFF
		self.gpu_id = 0
		if not self.slurm:
			self.world_size = torch.cuda.device_count()
			self.jobid = None
		else:
			self.world_size = int(os.environ["SLURM_NPROCS"])
			assert (self.world_size % torch.cuda.device_count()) == 0, "Use 1 task per GPU!"
			self.jobid = os.environ["SLURM_JOBID"]

		print(f"Training on {self.world_size} GPUs")

		# print project name and host name
		self.project_name = Path(__file__).parent.parent.basename()
		m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
		u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
		b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
		print(u_str + '\n' + m_str + '\n' + b_str)

		# project root
		self.project_root = Conf.OUT_PATH

		# set random seed
		self.seed = set_seed(seed)  # type: int

		# if the configuration file is not specified
		# try to load a configuration file based on the experiment name
		if not resume:
			tmp = Path(os.path.join(os.path.dirname(__file__), 'experiments', f"{self.exp_name}.json"))
			if conf_file_path is None and tmp.exists():
				conf_file_path = tmp
		else:
			tmp = Path(os.path.join(conf_file_path, 'configuration.json'))
			conf_file_path = tmp

		# read the JSON configuration file
		self.y = {}
		if conf_file_path is None:
			raise Exception(f"No model configuration file found {conf_file_path}")
		else:
			conf_file = open(conf_file_path, 'r')
			self.y = json.load(conf_file, cls=ConfigDecoder, object_hook=lambda d: DefaultNamespace(**d))

		# read configuration parameters from JSON file
		# or set their default value
		self.base_opts = self.y.get('experiment', {})  # type: dict
		self.epochs = self.base_opts.get('epochs', -1)  # type: int
		if self.device == 'cuda' and self.base_opts.get('device', None) is not None:
			self.device = self.base_opts.get('device')  # type: str
		self.val_epoch_step = self.base_opts.get('val_epoch_step', 1)  # type: int
		self.ck_epoch_step = self.base_opts.get('ck_epoch_step', 1)  # type: int
		self.finetune = self.base_opts.get("is_finetune", False)  # type: bool

		# define output paths
		if self.log_enabled:
			# Todo: be careful in multi-process!
			logdir = self.base_opts.get('logdir', '') # type: Path
			if logdir != '':
				logdir = Path(logdir)
				self.project_log_path = Path(logdir / 'log' / self.project_name)
			else:
				self.project_log_path = Path(Conf.OUT_PATH / 'log' / self.project_name)

			current_time = datetime.datetime.now(pytz.timezone("Europe/Rome"))
			if not resume:
				self.exp_full_name = f"{exp_name}.{current_time.year}.{current_time.month}.{current_time.day}.{current_time.hour}." \
						  f"{current_time.minute}.{current_time.second}.{get_unique_identifier()}"
			else:
				self.exp_full_name = conf_file_path.split('/')[-2]

			if not resume:
				self.exp_log_path = self.project_log_path / self.exp_full_name
				if not os.path.exists(self.exp_log_path):
					os.makedirs(self.exp_log_path, exist_ok=True)
			else:
				self.exp_log_path = conf_file_path.parent
		else:
			self.exp_full_name = conf_file_path.split('/')[-2]

		"""
		if self.world_size > 1:
			print(f"[WARNING]: Batch size is divided across {self.world_size} GPUs!")
			assert self.batch_size % self.world_size == 0
			self.batch_size = self.batch_size // self.world_size
		"""

		if data_root is not None:
			self.y.dataset.data_root = data_root

	def __getattr__(self, item):
		d = self.y.get(item, {})

		return d

	@property
	def is_cuda(self):
		# type: () -> bool
		"""
		:return: `True` if the required device is 'cuda'; `False` otherwise
		"""
		return 'cuda' in self.device

	def setup_device_id(self, rank):

		self.rank = rank
		self.local_rank = int(os.environ.get('LOCAL_RANK', -1))

		if self.slurm:
			self.gpu_id = rank % torch.cuda.device_count()  # Assuming an equal number of gpus per node
		else:
			self.gpu_id = rank

		if self.device == "cuda":
			self.device = f"cuda:{self.gpu_id}"
