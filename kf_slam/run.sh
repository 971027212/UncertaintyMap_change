#!/bin/sh

# 1. 允许本地连接 X Server（用于图形界面）
xhost +local:root

# 2. 【新增】先删除旧容器，防止报错 "The container name is already in use"
docker rm -f rosros_exploration 2>/dev/null

# 3. 启动容器
# 注意：加了 -d (后台运行) 和 -t (伪终端)，否则脚本会卡在这或者 exec 进不去
docker run -d -t \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    -v "$(pwd)":/root/ws/src/pioneer2dx/scripts/kf_slam/ \
    -p 11311:11311 \
    --name rosros_exploration \
    $1

# 4. 进入容器
docker exec -e "TERM=xterm-color" -it rosros_exploration bash
