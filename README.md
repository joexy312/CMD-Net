# CMD-Net
Implementation of &lt;Multi-prior Collaboration-based Model-driven Network for Accelerated MRI Reconstruction>

## requirements
-hydra-core
-torch
-numpy
-scipy
-matplotlib
-tqdm
-scikit-image

## Dataset
We use the data introduced in [VS-Net](https://github.com/j-duan/VS-Net), which can be downloaded at [GLOBUS](https://app.globus.org/file-manager?origin_id=15c7de28-a76b-11e9-821c-02b7a92d8e58&origin_path=%2F).
You can set your data directory in `cmdnet.yaml`.
Information of the FastMRI dataset can be found in [FastMRI](https://github.com/facebookresearch/fastMRI)

## Network training
run
```
python main.py
```

## E-mail
If you have any problem, please feel free to contact me: d210201019@stu.cqupt.edu.cn 
