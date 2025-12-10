.PHONY: install_uv install_evorun install_pixi

SHELL := /bin/bash
BINARY_PATH := /mnt/main0/shared/tools/evorun-0.1.0.tar.gz

all: install_evorun

install_uv:
	@source ~/.bashrc; \
	if ! command -v uv &> /dev/null; then \
		echo "Installing uv ..."; \
		"${SHELL}" <(curl -fsSL https://astral.sh/uv/install.sh); \
	else \
		echo "uv is already installed."; \
	fi

install_pixi:
	@source ~/.bashrc; \
	if ! command -v pixi &> /dev/null; then \
		echo "Installing pixi ..."; \
		export PIXI_VERSION=v0.54.0; \
		"${SHELL}" <(curl -fsSL https://pixi.sh/install.sh); \
	else \
		echo "Pixi is already installed."; \
	fi

install_fd:
	@source ~/.bashrc; \
	if ! command -v fd &> /dev/null; then \
		echo "Installing fd ..."; \
		sudo apt install fd-find; \
	else \
		echo "fd is already installed."; \
	fi

install_pipx:
	@source ~/.bashrc; \
	if ! command -v pipx &> /dev/null; then \
		echo "Installing pipx..."; \
		pixi global install pipx; \
	else \
		echo "pipx is already installed."; \
	fi

install: install_pixi install_uv install_pipx install_fd
	@source ~/.bashrc; \
	pipx install -f . \
	echo "slurm_run is installed globally"
