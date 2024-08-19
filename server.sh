#!/bin/bash
set -ev

if [ ! -e corridor_api ]; then
  echo "Please build and place corridor_api binary in this directory:" `pwd`
  exit -1
fi

export PORT=${PORT:="8080"}
export HOST=${HOST:="0.0.0.0"}

./corridor_api model/all_preprocessed_off_models_cgal model/asct-b-grlc.csv model/reference-organ-grlc.csv $HOST $PORT
