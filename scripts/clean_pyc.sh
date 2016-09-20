#!/bin/bash

find . | grep '\.pyc$' | awk '{print "remove "$0;system("rm -f "$0)}'