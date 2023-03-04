

if ! [ -f "ILSVRC2012_img_train.tar" ]; then
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
fi
if ! [ -f "ILSVRC2012_img_val.tar" ]; then
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
fi
if ! [ -f "ILSVRC2012_devkit_t12.tar.gz" ]; then
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
fi
if ! [ -f "ILSVRC2012_img_test_v10102019.tar" ]; then
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
fi

# extract train_tar file
mkdir train/
tar xf ILSVRC2012_img_train.tar -C train/

cd train/
all_files=`ls -1 *.tar`
for file in $all_files; do
    echo "Extracting $file"
    tar -xf $file --one-top-level=${file%.tar}
    rm $file
done
cd ../

# extract val_tar file
mkdir val/
tar xf ILSVRC2012_img_val.tar -C val/
cd val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ../


tar xfz ILSVRC2012_devkit_t12.tar.gz
