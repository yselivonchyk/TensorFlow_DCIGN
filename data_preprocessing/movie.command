#!/bin/bash
mencoder "mf://*.jpg" -mf fps=5 -o out.avi -ovc lavc -lavcopts vco dec=msmpeg4v2:vbitrate=640
