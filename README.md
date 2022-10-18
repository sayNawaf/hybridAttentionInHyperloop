# hybridAttentionInHyperloop

HYPERSPECTRAL IMAGE
are images in which one continuous spectrum is measured for each pixel. Normally, the spectral resolution is given in nanometers or wave numbers
Hyperspectral images can be obtained from many different electromagnetic measurements. The most popular are visible (VIS), NIR, middle infrared (MIR), and Raman spectroscopy. Nevertheless, there are many other types of HSI that are gaining popularity like confocal laser microscopy scanners that are able to measure the complete emission spectrum at certain excitation wavelength for each pixel, Terahertz spectroscopy, X-ray spectroscopy, 3D ultrasound imaging, or even magnetic resonance.

HYPERLOOP
proposes a self-looping convolution network for more efficient HSI classification. In this method, each layer is a self-looping block
contains both forward and backward connections, which means that each layer is the input and the output of
every other layer, thus forming a loop. These loopy connections within the network allow for maximum information flow, thereby giving us a high level feature extraction. The self-looping connections enable us to efficiently control the network parameters, further allowing us to go for a wider architecture with a multiscale
setting, thus giving us abstract representation at different spatial levels..
PAPER CITATION:https://www.researchgate.net/publication/356812988_HyperLoopNet_Hyperspectral_image_classification_using_multiscale_self-looping_convolutional_networks

HYBRID ATTENTION.
contain spatial attention  and channel attention
PAPER CITATION:https://pubmed.ncbi.nlm.nih.gov/35479608/

Spaital Attention
spatial attention module pays more
attention to the spatial position information of the discriminant part, which is a supplement to the channel attention

Hybrid Attention
(e convolutional characteristic maps produced by the characteristic functions contain different characteristic channels, and in the fine-grained
picture recognition problem, each characteristic channel may
represent different information in the picture, some of which
contain irrelevant picture background information and are
redundant. 

My Implementation.
implmented hyperloopNET from the paper and integrated HYBRID attention in it.
