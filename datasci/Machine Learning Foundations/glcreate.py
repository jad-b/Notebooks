import graphlab as gl
GLCREATE_PRODUCT_KEY = 'E14F-47F4-8873-A9D7-EFB6-22EE-63F7-50BB'
gl.product_key.set_product_key(GLCREATE_PRODUCT_KEY)
# Display GL canvas in notebook
gl.canvas.set_target('ipynb')
# Number of workers
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 16)