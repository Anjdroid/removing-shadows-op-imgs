# Removing shadows from images

A pipeline for shadow detection and removal.

## Prerequisites

- python version >= 3.6

## How to setup and run

- download zip or clone the project
- go into folder 

### install required packages

$ pip install -r requirements.txt
  
- for shadow removal with default image:
```
python remove_shadows.py
```
- for shadow removal with specified image and visualisation of results:
```
python remove_shadows.py --image_path --visualize 1
```
