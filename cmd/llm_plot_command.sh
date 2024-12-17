inspect_args

model=${args[model]}

if [ -z "$tag" ]; then
    model="latest"
fi

./cray build-image

declare -a plot_command_parts
plot_command_parts=(
      "python" "/app/cray/sdk/cli/main.py" "plot" "--model" "$model"
)

plot_command="${plot_command_parts[*]}"

echo $command

declare -a docker_command_parts

docker_command_parts=("docker" "run" "--rm")

docker_command_parts+=("cray:latest" "sh" "-c" "'$plot_command'")

docker_command="${docker_command_parts[*]}"
echo $docker_command
eval $docker_command

