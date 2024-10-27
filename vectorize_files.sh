#!/bin/bash
data_path=${data_path:-data}
vector_path=${vector_path:-vector_storage}

while [ $# -gt 0 ]; do
	if [[ $1 == *"--"* ]]; then
		param="${1/--/}"
		declare $param="$2"
		echo $1 $2 #Optional to see the parameter:value result
	fi
	shift
done

echo ${data_path}
echo ${vector_path}

python src/vectorize_data.py --knowledge_dir ${data_path} --vector_storage_path ${vector_path}