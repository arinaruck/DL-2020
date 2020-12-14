#DATASET="edges2handbags.tar.gz"
DATASET="facades.tar.gz"
wget "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/${DATASET}"
tar -xzf $DATASET
