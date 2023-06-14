#!/usr/bin/env bash
ps -ef|grep generate_chat_web.py|awk '{print $2}'| xargs kill -9