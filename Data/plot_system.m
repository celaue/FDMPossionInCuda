table = readtable("system.csv");
x = table.x;
y = table.y;
[X,Y] = meshgrid(x,y);
Z = griddata(x,y,table.value,X,Y);
surf(X,Y,Z);
xlabel("x axis");
ylabel("y axis");
