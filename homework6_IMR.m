r = 0.028;
    d = 0.149;
dphi_r = diff(rightEncoder); %Number of rotations per time measurement
dphi_l = diff(leftencoder); %Number of rotations per time measurement
ds = r/2*(dphi_r + dphi_l);
dtheta = r/d*(dphi_r - dphi_l);

x_od = zeros(size(rightEncoder,1),1);
y_od = zeros(size(rightEncoder,1),1);
theta = zeros(size(rightEncoder,1),1);

for i = 2:size(rightEncoder,1)
    x_od(i) = x_od(i-1) + ds(i-1)*sind(theta(i-1));
    y_od(i) = y_od(i-1) + ds(i-1)*cosd(theta(i-1));
    theta(i) = theta(i-1) + dtheta(i-1);
end


%x_od = gnegate(x_od);
 plot(x_od, y_od);
%% left wall 
 x_l = zeros(size(rightEncoder,1),1);
 y_l = zeros(size(rightEncoder,1),1);
 
 for i=1:size(rightEncoder,1)
     x_l(i) = x_od(i)+ leftUtrasonic(i)*cosd(theta(i)-90);
     y_l(i) = y_od(i)+ leftUtrasonic(i)*cosd(theta(i)-90);
 end
 
 hold on;
 scatter(x_l,y_l,25,'.');
 
 %% right wall 
 x_r = zeros(size(rightEncoder,1),1);
 y_r = zeros(size(rightEncoder,1),1);
 
 for i=1:size(rightEncoder,1)
     x_r(i) = x_od(i)- rightUtralsoni(i)*cosd(theta(i)+90);
     y_r(i) = y_od(i)- rightUtralsoni(i)*cosd(theta(i)+90);
 end
 
 hold on;
 scatter(x_r,y_r,25,'.');

