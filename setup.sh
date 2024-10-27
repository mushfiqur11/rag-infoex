#!/bin/bash
task=${task:-install_rag_pipeline}
host=${host:-own}
llm=${llm:-microsoft/Phi-3-mini-4k-instruct}

while [ $# -gt 0 ]; do

	if [[ $1 == *"--"* ]]; then
		param="${1/--/}"
		declare $param="$2"
		echo $1 $2 #Optional to see the parameter:value result
	fi

	shift
done

if [[ "$host" = "hopper" ]]; then
	echo "Installing virtual env on hopper"
	if [ "$task" = "install_rag_pipeline" ] || [ "$task" = "all" ]; then
		module load gnu10/10.3.0-ya
		module avail python
		module load python/3.10.1-5r
		module load git
		rm -rfd vnv/info_ex
		python -m venv vnv/info_ex
		source vnv/info_ex/bin/activate
		pip install --upgrade pip
		module unload python
		pip3 install torch==2.4.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
		pip install -r requirements.txt
	fi
fi

if [[ "$host" = "own" ]]; then
	echo "Installing system requirements and env"
	if [ "$task" = "install_rag_pipeline" ] || [ "$task" = "all" ]; then
		rm -rfd vnv/info_ex
		python -m venv vnv/info_ex
		source vnv/info_ex/bin/activate
		pip install --upgrade pip
		module unload python
		pip3 install torch==2.4.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
		pip install -r requirements.txt
	fi
fi

if [ "$task" = "download_llm" ] || [ "$task" = "all" ]; then
	python src/download_llm.py --model_id ${llm}
fi

