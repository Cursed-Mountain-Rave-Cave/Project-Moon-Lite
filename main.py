import numpy as np
from matplotlib import pyplot
from tqdm import trange

r = 0.1
d = 1
n = 100

it_count = 10000
eps = 1e-4
dif = 0

x = np.linspace(-d/2, d/2, n+1)
y = np.linspace(-d/2, d/2, n+1)

xv, yv = np.meshgrid(x, y)

inner_mask = (xv**2 + yv**2 <= r**2)
# inner_mask = (np.abs(xv) < r) 

f = 15.*inner_mask

sub_mask = inner_mask[1:-1, 1:-1]
for i in trange(it_count):
    df = (
        f[:-2,1:-1] 
        + f[2:,1:-1] 
        + f[1:-1,:-2] 
        + f[1:-1,2:]
        - 4 * f[1:-1, 1:-1]
    ) / 4 * (sub_mask == 0)
    
    dif = np.abs(df).max()
    f[1:-1,1:-1] += df

    if dif < eps:
        break

print('MAE:', dif)

figure = pyplot.figure(figsize=(6, 6))
pyplot.imshow(df)
pyplot.axis('equal')
pyplot.xticks(*zip(*[*enumerate(x.round(3))][::n//5]))
pyplot.yticks(*zip(*[*enumerate(y.round(3))][::n//5]))
pyplot.colorbar()
figure.tight_layout()
figure = pyplot.figure(figsize=(6, 6))
pyplot.imshow(f)
pyplot.axis('equal')
pyplot.xticks(*zip(*[*enumerate(x.round(3))][::n//5]))
pyplot.yticks(*zip(*[*enumerate(y.round(3))][::n//5]))
pyplot.colorbar()
figure.tight_layout()
figure = pyplot.figure(figsize=(6, 6))
pyplot.imshow(inner_mask)
pyplot.axis('equal')
pyplot.xticks(*zip(*[*enumerate(x.round(3))][::n//5]))
pyplot.yticks(*zip(*[*enumerate(y.round(3))][::n//5]))
pyplot.colorbar()
figure.tight_layout()
pyplot.show()

'''
(
    (uij+1 - uij) / (xij+1 - xij) - (uij - uij-1) / (xij - xij-1)
) / (
    (xij+1 - xij-1) / 2
)
+
(
    (ui+1j - uij) / (yi+1j - yij) - (uij - ui-1j) / (yij - yi-1j)
) / (
    (yi+1j - yi-1j) / 2
)
= 0


  (yi+1j - yi-1j) * (yi+1j - yij) * (yij - yi-1j) * (xij - xij-1) * (uij+1 - uij) 
- (yi+1j - yi-1j) * (yi+1j - yij) * (yij - yi-1j) * (xij+1 - xij) * (uij - uij-1)
+ (xij+1 - xij-1) * (xij+1 - xij) * (xij - xij-1) * (yij - yi-1j) * (ui+1j - uij) 
- (xij+1 - xij-1) * (xij+1 - xij) * (xij - xij-1) * (yi+1j - yij) * (uij - ui-1j)
= 0

uij = (
    (yi+1j - yi-1j) * (yi+1j - yij) * (yij - yi-1j) * (
        (xij - xij-1) * uij+1 
        + (xij+1 - xij) * uij-1
    ) 
    + (xij+1 - xij-1) * (xij+1 - xij) * (xij - xij-1) * (
        (yij - yi-1j) * ui+1j 
        + (yi+1j - yij) * ui-1j
    )
) / (
      (xij+1 - xij-1) * (xij+1 - xij) * (xij - xij-1) * (yi+1j - yi-1j)
    + (yi+1j - yi-1j) * (yi+1j - yij) * (yij - yi-1j) * (xij+1 - xij-1)
) 
'''