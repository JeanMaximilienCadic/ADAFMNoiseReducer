#!/usr/bin/env bash

#Variables
VERSION="1.0.2"
WHL_NAME="adafmnoisereducer-$VERSION-py3-none-any.whl";
WHL_NAME_NEW="adafmnoisereducer-$VERSION-py36-none-any.whl";

##Build wheel
COMMAND="rm dist/*.whl"; echo $COMMAND;  $COMMAND
python setup.py  bdist_wheel --python-tag py36

##Clean
COMMAND="mv dist/*.whl $WHL_NAME_NEW"; echo $COMMAND;  $COMMAND
COMMAND="rm -r dist"; echo $COMMAND;  $COMMAND

#Upload
COMMAND="twine upload $WHL_NAME_NEW"; echo $COMMAND;  $COMMAND
