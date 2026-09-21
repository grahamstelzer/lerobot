#!/usr/bin/env bash

# meant to literally follow https://huggingface.co/docs/lerobot/en/installation
# for all the steps to get so101 completely up and running on linux setup with lerobot
# ...
# as well as make it easier to run single commands without filling in all the fields manually
# ex. lerobot-train (many args)... -> ./so101.sh --train-only

# i python scripts and docker container could also accomplish the same thing probably.
# anyways.



# exit on errors
set -euo pipefail
# -e: exit on command returns nonzero status
# -u: exit on unset variables
# -o pipefail: exit if any pipelien command fails

export LANG=en_US.UTF-8 # funny emojis

# check for .env
if [[ ! -f .env ]]; then
    echo "error .env not found"
    exit 1
fi



# load all variables from .env file into the current shell environment
set -a
source .env
set +a 



# local variables, set all to TRUE, input parser will handle what is actually run
run_software_setup=true
run_hardware_setup=true
run_teleop=true
run_camera_setup=true



# CONDA ACTIVATION MUST HAPPEN EVERY TIME WE USE LEROBOT COMMANDS
# therefore, i am placing this at the top to run every time the script is run
# check conda (miniforge) installation:

# NOTE: user prompt likely redundant but gives user a chance to match conda
#       type with the one suggested by lerobot installtion docs

if [[ -z "$CONDA_INSTALLED" ]] || [[ "$CONDA_INSTALLED" != "true" ]]; then
    read -p "is conda (miniforge) installed? (y/n): " conda_installed # TODO: more elegant flag/prompt combo?
    if [[ "$conda_installed" == "y" ]]; then
        echo "    $(echo -e '\u2714') following commands will assume conda is installed correctly"
        sed -i "s|^CONDA_INSTALLED=.*|CONDA_INSTALLED=true|" .env
    else
        # install conda (miniforge)
        wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash Miniforge3-$(uname)-$(uname -m).sh
        sed -i "s|^CONDA_INSTALLED=.*|CONDA_INSTALLED=true|" .env
    fi
fi

# check if conda env for lerobot exists 
echo "checking for pre-existing lerobot env..."
if conda env list | grep -q "^lerobot\s"; then
    echo "    $(echo -e '\u2714') lerobot conda environment already exists"
else
    echo "lerobot conda environment not found. creating new environment..."
    conda create -n lerobot python=3.10 -y
fi


eval "$(conda shell.bash hook)"
conda activate lerobot


# setup ffmpeg and lerobot
software_setup() {

    echo "running software setup"

    read -p "is ffmpeg vs 7.1.1 installed? (y/n) " ffmpeg_installed
    if [[ "$ffmpeg_installed" == "y" ]]; then
        echo "    $(echo -e '\u2714') ffmpeg installed"
    else
        conda install ffmpeg -c conda-forge
    fi


    # check lerobot installation
    read -p "is lerobot already installed with appropriate dependencies? (y/n): " lerobot_installed
    if [[ "$lerobot_installed" == "y" ]]; then
        echo "    $(echo -e '\u2714') following commands will assume lerobot is installed with correct dependencies. installation docs: https://huggingface.co/docs/lerobot/en/installation"
    else
        # install lerobot and dependencies
        # (note) installs from source
        echo "    installing lerobot from source into current location"
        git clone https://github.com/huggingface/lerobot.git
        cd lerobot
        echo "    installing lerobot dependencies"
        pip install -e .

        # TODO: add extras/choices for models to install?
        #       caution: i think they might have different deps though, maybe need different conda envs

        # kinda hacky, oversight by me, this script is meant to be run inside of lerobot
        # TODO: add filepath lerobot/ to the beginning of other filepaths?
        cd ..
        cp so101.sh lerobot/
        rm so101.sh
        cd lerobot
    fi
}



# at this point the software should be setup and we will now run hardware checks
hardware_setup() {

    echo "running hardware setup"
    echo "MAKE SURE FOLLOWER AND TELEOP ARE PLUGGED INTO THE PC"

    read -p "attempt to find follower port? (y/n): "  find_follower_port
    if [[ "$find_follower_port" == "y" ]]; then
        # find port of follower
        lerobot-find-port
        read -p "enter the port of the follower: " new_follower_port
        # update .env file with the new follower port
        sed -i "s|^FOLLOWER_PORT=.*|FOLLOWER_PORT=$new_follower_port|" .env
        echo "    $(echo -e '\u2714') FOLLOWER_PORT updated to $new_follower_port in .env"
    fi


    read -p "attempt to find teleop port? (y/n): " find_teleop_port
    if [[ "$find_teleop_port" == "y" ]]; then
        # find port of teleop
        lerobot-find-port
        read -p "enter the port of the teleop: " new_teleop_port
        # update .env file with the new teleop port
        sed -i "s|^TELEOP_PORT=.*|TELEOP_PORT=$new_teleop_port|" .env
        echo "    $(echo -e '\u2714') TELEOP_PORT updated to $new_teleop_port in .env"
    fi


    # set follower and teleop names (used to set configs in data collection/training commands)
    read -p "set names of follower and teleop? (y/n): " set_names
    if [[ "$set_names" == "y" ]]; then
        read -p "set name of follower: " new_follower_name
        sed -i "s|^FOLLOWER_NAME=.*|FOLLOWER_NAME=$new_follower_name|" .env
        read -p "set name of teleop: " new_teleop_name
        sed -i "s|^TELEOP_NAME=.*|TELEOP_NAME=$new_teleop_name|" .env
    fi

    # run follower calibration
    read -p "run follower calibration? (y/n): " run_follower_calibration
    if [[ "$run_follower_calibration" == "y" ]]; then
        echo "running follower calibration:"
        lerobot-calibrate \
            --robot.type=so101_follower \
            --robot.port=$FOLLOWER_PORT \
            --robot.id=$FOLLOWER_NAME
    fi

    # run teleop calibration
    read -p "run teleop calibration? (y/n): " run_teleop_calibration
    if [[ "$run_teleop_calibration" == "y" ]]; then
        echo "running teleop calibration:"
        lerobot-calibrate \
            --teleop.type=so101_leader \
            --teleop.port=$TELEOP_PORT \
            --teleop.id=$TELEOP_NAME
    fi
}



camera_setup() {

    echo "running camera setup"

    lerobot-find-cameras opencv
    echo "found following images (saved in /outputs/captured_images):"
    ls -la ./outputs/captured_images

    local camera_index=1
    while true; do
        read -p "enter port for camera $camera_index like /dev/videoX (enter 'q' to quit): " camera_port
        if [[ "$camera_port" == "q" ]]; then
            break
        fi
        sed -i "s|^CAMERA_${camera_index}_PORT=.*|CAMERA_${camera_index}_PORT=$camera_port|" .env
        echo "    $(echo -e '\u2714') CAMERA_${camera_index}_PORT updated to $camera_port in .env"
        ((camera_index++))
    done
}



teleop() {

    echo "running teleop"
    echo "NOTE: assumes cameras already setup"

    # teleop with camera:
    # NOTE: only working with 2 cameras right now....
    lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_NAME \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: $CAMERA_1_PORT, width: 640, height: 480, fps: 30}, \
                       camera2: {type: opencv, index_or_path: $CAMERA_2_PORT, width: 640, height: 480, fps: 30}, \
                       camera3: {type: opencv, index_or_path: $CAMERA_3_PORT, width: 640, height: 480, fps: 30}, \
                       camera4: {type: opencv, index_or_path: $CAMERA_4_PORT, width: 640, height: 480, fps: 30}, \
                       }" \
    --teleop.type=so101_leader \
    --teleop.port=$TELEOP_PORT \
    --teleop.id=$TELEOP_NAME \
    --display_data=true





data_collection() {


    # TODO: add env var for current dataset repo?


    # hf login for repo id (auto push? change to push afterwards)
    hf auth login --token $HF_TOKEN

    # vars
    # num episodes, name of repo, hf user?? (add to env probably), task string
    # NOTE: must make sure pathing is correct? uses cache
    read -p "enter number of episodes: " num_eps
    echo $num_eps

    read -p "enter name of local repo: " local_repo_name
    echo $local_repo_name

    read -p "enter name of public repo (will be shown on huggingface)" hf_repo_name
    echo $hf_repo_name
    
    read -p "enter prompt of task: " task_prompt
    echo $task_prompt


    # command
    lerobot-record \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_NAME \
    --teleop.port=$TELEOP_PORT \
    --teleop_id=$TELEOP_NAME \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: $CAMERA_1_PORT, width: 640, height: 480, fps: 30}, \
                       camera2: {type: opencv, index_or_path: $CAMERA_2_PORT, width: 640, height: 480, fps: 30} }" \
    --display_data=false \
    --dataset.repo_id=grahamwichh/eval_${hf_repo_name} \
    --dataset.single_task="${task_prompt}" \
    --policy.path=grahamwichh/$local_repo_name


    echo ""

}



train() {

    echo ""
}


replay() {


    # TODO: add env var for current dataset repo?


    # lerobot-replay \
    # --robot.type=so101_follower \
    # --robot.port=$FOLLOWER_PORT \
    # --robot.id=$FOLLOWER_NAME \
    # --dataset.repo_id=...
    echo ""

}



inference() {

    # TODO: must handle deletion of repo each time record command is run
    # TODO: add ways to download from huggingface (pi0, pi0.5... etc.)


    echo ""
}






# input parsing loop
# NOTE1: while loop condition parses loop while there is an unprocessed arg
#       "shift" at the end of the loop literally shifts the current argument being looked at in $1 variable
# NOTE2: assumes everything is turned on and will run (=true) and turns off items as specified by args 
while [[ $# -gt 0 ]]; do
    case "$1" in # check current first arg, shifts at end of loop iter
        --software-setup-only)
            # turn off other things
            echo "only running software setup"
            run_hardware_setup=false
            run_software_setup=true # redundant, but for readability
            run_teleop=false
            run_camera_setup=false
        ;;
        --hardware-setup-only)
            # turn off other things
            echo "only running hardware setup"
            run_hardware_setup=true # redundant, but for readability
            run_software_setup=false
            run_teleop=false 
            run_camera_setup=false
        ;;
        --camera-setup-only)
            # turon off everything else
            echo "only running camera setup"
            run_hardware_setup=false
            run_software_setup=false
            run_teleop=false
            run_camera_setup=true # redundant, but for readability
        ;;
        --teleop-only)
            # turn off eveything else
            echo "only running teleop"
            run_hardware_setup=false
            run_software_setup=false
            run_teleop=true # redundant, but for readability
            run_camera_setup=false
        ;;    # lerobot offers way to replay episodes of dataset:
    -h|--help)
        echo "Usage: ./${SCRIPT_NAME} [ --software-setup-only | ...]"
        # TOOD: add all options
        exit 0
        ;;
    *)
        echo "[ERROR] Unknown flag: $1" >&2
        exit 1
    esac
    shift
done



if [[ "$run_software_setup" == true ]]; then
    software_setup
fi
if [[ "$run_hardware_setup" == true ]]; then
    hardware_setup
fi
if [[ "$run_camera_setup" == true ]]; then
    camera_setup
fi
if [[ "$run_teleop" == true ]]; then
    teleop
fi