The 4 .ipynbs are autoencoders trained on the IBIS data to impute either cortical thickness (CT) or surface area (SA).
These are done to impute the 1 year data FROM the 2 year data and vice versa.

All CTSA 2-1yr has all Twins, CONTEs from Gilmore, and IBIS data that appears with both 1 and 2 year data points.
It is used to extract the IBIS data for autoencoder training. And later the Gilmore CONTEs for calculating the MAE

MAE is calculated by inverse_transforming the predicted Gilmore CONTEs and then doing the normal MAE calculation.
Only able to be performed on those with both 1 and 2yr in the Interpolated Gilmore Data for MAE folder.
MAEs
Gilmore CT1y - -0.00572973
Gilmore CT2y - -0.473055607
Gilmore SA1y - 21.25061733
Gilmore SA2y - -95.74517168


