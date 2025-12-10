.PHONY: install_uv install install_pixi install_fd_find link_fd install_pipx
.ONESHELL:

SHELL := /bin/bash

all: install

install_uv:
	@source ~/.bashrc;
	if ! command -v uv &> /dev/null; then
		echo "Installing uv ...";
		"${SHELL}" <(curl -fsSL https://astral.sh/uv/install.sh);
	else
		echo "uv is already installed.";
	fi

install_pixi:
	@source ~/.bashrc;
	if ! command -v pixi &> /dev/null; then
		echo "Installing pixi ...";
		export PIXI_VERSION=v0.54.0;
		"${SHELL}" <(curl -fsSL https://pixi.sh/install.sh);
	else
		echo "Pixi is already installed.";
	fi

install_fd_find:
	@if ! command -v fdfind &> /dev/null; then
		echo "Installing fd ...";
		sudo apt install fd-find;
	else
		echo "fdfind is already installed.";
	fi

link_fd: install_fd_find
	@mkdir -p "$(HOME)/.local/bin"
	if ! [ -f $(HOME)/.local/bin/fd ]; then
		ln -s $$(command -v fdfind) $(HOME)/.local/bin/fd;
	fi

install_pipx:
	@source ~/.bashrc;
	if ! command -v pipx &> /dev/null; then
		echo "Installing pipx...";
		~/.pixi/bin/pixi global install pipx;
	else
		echo "pipx is already installed.";
	fi

install: install_pixi install_uv install_pipx link_fd
	@source ~/.bashrc;
	pipx install -f .;
	echo "slurm_run is installed globally";
