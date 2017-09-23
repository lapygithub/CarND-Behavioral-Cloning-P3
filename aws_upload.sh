tar -jcvf ../carnd_bc.tar.bz2 /Users/mikel/code/CarND-Behavioral-Cloning-P3
scp -i ~/.aws/lapy_vpc_jupyter_kp_20170823.pem ../carnd_bc.tar.bz2  ec2-user@ec2-52-37-143-190.compute-1.amazonaws.com:~
