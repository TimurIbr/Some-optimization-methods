This is output of Optimyzed Dual triangle method and Dual triangle method on sample.

Results are compared with minimize(method='BFGS').

Sample:
  functtion :
  $$ ||Ax - b||_{2}^{2} - \\lambda_{min} (A^{T}A) \\cdot \\frac{1}{2}||x||_{2}^{2}$$
  A = np.array([[1, 2], [2, 3]])
  b = np.array([1, 2])
  
Results after 1000 steps:

  Dual triangle method (1,72066430786	-0,444433854843) .
  
  Optimyzed Dual triangle method (1,72403238111	-0,446516173029) .
  
  minimize(method='BFGS') (1,72403722752	-0,446518440485) .

