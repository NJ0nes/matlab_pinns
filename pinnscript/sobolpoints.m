function points = sobolpoints(n)

pointSet = sobolset(2);
points = net(pointSet,n);

end