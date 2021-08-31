from hilbertcurve.hilbertcurve import HilbertCurve
ite=4; 
dim=3;
hilbert_curve = HilbertCurve(ite, dim)
distances = list(range(4096))
points = hilbert_curve.points_from_distances(distances)
print(points)
for point, dist in zip(points, distances):
    print(point, dist)
