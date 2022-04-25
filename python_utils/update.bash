#!/bin/bash

while true
do
	rosservice call /meshserv test
	sleep 1
done
