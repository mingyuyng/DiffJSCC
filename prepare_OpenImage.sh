


mkdir data
cd data

mkdir OpenImage
cd OpenImage

aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_0.tar.gz ./
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_1.tar.gz ./
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_2.tar.gz ./

tar -zxvf train_0.tar.gz
tar -zxvf train_1.tar.gz
tar -zxvf train_2.tar.gz

rm train_0.tar.gz
rm train_1.tar.gz
rm train_2.tar.gz

mv train_0/* ./
mv train_1/* ./
mv train_2/* ./

rm -r train_0/
rm -r train_1/
rm -r train_2/

echo "Finished"
