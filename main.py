# -*- coding: utf-8 -*-
# ---------------------

import torch
import logging
from conf import Conf

import click
import torch.backends.cudnn as cudnn

from trainer import trainer_run

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--conf_file_path', type=str, default=None)
@click.option('--seed', type=int, default=None)
def main(exp_name, conf_file_path, seed):
	# type: (str, str, int) -> None

	assert torch.backends.cudnn.enabled, "Running without cuDNN is discouraged"

	# if `exp_name` is None,
	# ask the user to enter it
	if exp_name is None:
		exp_name = input('>> experiment name: ')

	# if `exp_name` contains '!',
	# `log_each_step` becomes `False`
	log_each_step = True
	if '!' in exp_name:
		exp_name = exp_name.replace('!', '')
		log_each_step = False

	# if `exp_name` contains a '@' character,
	# the number following '@' is considered as
	# the desired random seed for the experiment
	split = exp_name.split('@')
	if len(split) == 2:
		seed = int(split[1])
		exp_name = split[0]

	cnf = Conf(conf_file_path=conf_file_path, seed=seed,
			   exp_name=exp_name, log_each_step=log_each_step)
	print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

	# Setup logging
	logging.basicConfig(
		format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
		level=logging.INFO,
	)

	cnf_attrs = vars(cnf)
	for k in cnf_attrs:
		s = f'{k} : {cnf_attrs[k]}'
		logging.info(s)

	# Run training
	trainer_run(cnf)

if __name__ == '__main__':
	main()
