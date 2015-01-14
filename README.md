histogram_equalization
======================

- Equalizes the contrast of an image using a histogram
- Spreads out the contrast
- Lighter areas become darker and darker areas become lighter
- Makes it easier to identify features in an image
- A cummulative distribution function is used to find the new gray value of a pixel
- Usage: `python histogram_equalization.py -i <inputf> [-o <outputf> -t <transferf> -a <histinf> -b <histoutf>]`
- `-o` specifies output image path, `-t` specifies transfer function plot path, `-a` specifies input histogram path, `-b` specifies output histogram path

Input Image

![alt tag](http://i.imgur.com/oVq73eA.png)

Output Image (after histogram equalization)

![alt tag](http://i.imgur.com/6t3ZsuV.png)

Transfer Function

![alt tag](http://i.imgur.com/bAcLTlg.png)

Input Histogram

![alt tag](http://i.imgur.com/K85W2h4.png)

Output Histogram

![alt tag](http://i.imgur.com/HLUr7QP.png)
