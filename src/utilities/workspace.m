clear;
show_robot_move = true; % For show robot with peter corke
% Proximal and distal length
% If distal > proximal, there are not type 2 sigularities

% NOTE: In the paper said: Clearly, if the proximal and distal links of a
% five-bar robot are of different lengths, the robot’s workspace will have
% holes and it would be impossible to beat the RP-5AH design.
% Therefore, all links should be of the same length, l 1 = l 2 = l.
% I cannot understand what it means.
proximal = 0.230;
distal = 0.350;
d =0.0; % Distance between bases. The smaller d is, the larger workspace become
base = d/2; % Absolute distance from the reference frame to base link

d1 = [-base, proximal, distal];
d2 = [base, proximal, distal];
q1 = [0, 0, 0]; % [0.4072, 0, 0];
q2 = [0, 0, 0]; % [1.1393, 0, 0];
ass_mode = 1; % -1 for 2nd model

j=0;
dq=0.1;
qa_ =-pi:dq:pi;
qb_=-pi:dq:pi;
max_j = 15876;
pp_ = zeros(2,max_j);
q1_ = zeros(3,max_j);
q2_ = zeros(2,max_j);

for m=1:length(qa_)
        q2(1)=qa_(m);
    for n=1:length(qb_)
        q1(1)=qb_(n);
        % from dextar paper
        A12 = [d1(2)*cos(q1(1)); d1(2)*sin(q1(1))];
        A22 = [d2(2)*cos(q2(1)); d2(2)*sin(q2(1))];
%         s = sqrt(sum((A22-A12).^2));
        s = norm(A22-A12);
        if s <= 2 * distal
            j=j+1;
            %from dextar paper
            v = (A22-A12)/2;
            h = [-v(2);  v(1)]/s * sqrt(4 * proximal^2 - s^2);
            x = A12(1) + v(1) + ass_mode * h(1);
            y = A12(2) + v(2) + ass_mode * h(2);
            x22 = A22(1) + v(1) + ass_mode * h(1);
            y22 = A22(2) + v(2) + ass_mode * h(2);
%             if round(x22) ~= round(x) || round(y22) ~= round(y)
%                 disp('Fractura')
%                 break
%             end
            pp_(:,j)=[x;y];

            % for q12 and q22 from
            A1C = [x;y] - A12;
            A2C = [x;y] - A22;
            q1(2) = atan2(A1C(2), A1C(1)) - q1(1);
            q2(2) = atan2(A2C(2), A2C(1)) - q2(1);
%             q1(2) = atan2(y-d1(2)*sin(q1(1)), x-d1(2)*cos(q1(1))) - q1(1);
%             q2(2) = atan2(y-d2(2)*sin(q2(1)), x-d2(2)*cos(q2(1))) - q2(1);
            if isnan(q1(2)) || isnan(q2(2))
                q1(2) = 0;
                q2(2) = 0;
            end
            q1(3) = q2(1) + q2(2) - q1(1) - q1(2);
            q1_(:,j)= [q1(1); q1(2); q1(3)];
            q2_(:,j)= [q2(1); q2(2)];
        else
        end
    end
end
%% Show how the robots move
DH = [  1, 0, proximal, 0, 0;
        2, 0, distal, 0, 0;
        3, 0, 0, 0, 0;];
DH2 =  [4, 0, proximal, 0, 0;
        5, 0, distal, 0, 0;
        6, 0, 0, 0, 0;];
arm1 = SerialLink(DH, 'name', 'arm1');
arm1.base = transl(d1(1), 0, 0);
arm2 = SerialLink(DH2, 'name', 'arm2');
arm2.base = transl(d2(1), 0, 0);
if show_robot_move
    for i=1:10:length(q1_)
        arm1.plot(q1_(:,i)')
        hold on;
        arm2.plot([q2_(:,i)',0])
    end
end
close all;
%% Joint Space
figure(1)
plot(q1_(1,:),q2_(1,:),'b.')
title('Joint Space')

dtheta = 0.09; % Variable for fix ranges
%% Type 1 singularity
% Are 2 circles of radius distal - proximal and 2 arcs
% of circles of radius distal+proximal
figure(2)
plot((proximal + distal)*cos(dtheta-pi/2:0.01:pi/2-dtheta)-base, ...
    (proximal + distal)*sin(dtheta-pi/2:0.01:pi/2-dtheta),'k--')
hold on;
plot(pp_(1,:),pp_(2,:),'g.');
plot((proximal + distal)*cos(dtheta+pi/2:0.01:-dtheta+3*pi/2)+base, ...
    (proximal + distal)*sin(dtheta-pi/2:0.01:-dtheta+pi/2),'k--');
% This two circles appear when when distal length is bigger than proximal length
plot((distal-proximal)*cos(0:0.01:2*pi)+base, ...
    (distal-proximal)*sin(0:0.01:2*pi),'k--');
plot((distal-proximal)*cos(0:0.01:2*pi)-base, ...
    (distal-proximal)*sin(0:0.01:2*pi),'k--');

%% Type 2
% singularity loci for all working modes consist of two
% circles of radius distal described by point C when A12 and A22 coin-
% cide (denoted by C_1 and C_2 ) and a sextic (denoted by S:).
% The parametric equations for the two Type 2 singularity
% circles with center [0, proximal ...]
% Note: I believe that type 2 singularity disappear when distal length is
% bigger than proximal length

% Circles for find center of type 2 singularity
% plot(proximal*cos(0:0.01:2*pi)+base, ...
%     proximal*sin(0:0.01:2*pi),'m');
% plot(proximal*cos(0:0.01:2*pi)-base, ...
%     proximal*sin(0:0.01:2*pi),'m');
if proximal >= distal
    % Assembly 1
    a = 0.31+pi/2; % Start point
    x_1 = distal*cos(-a:0.01:3*pi/2+0.4);
    y_1 = 0.5 * sqrt(4*proximal^2-d^2) + distal*sin(-a:0.01:3*pi/2+0.4) ;
    plot(x_1, y_1,'r')
    % plot(distal*-cos(a:0.01:3*pi/2-0.4), -0.5 * sqrt(4*distal^2-d^2) + ...
    %     distal*-sin(a:0.01:3*pi/2-0.4),'r')

    % Assembly -1
    x_2 = distal*cos(0:0.01:2*pi);
    y_2 = -0.5 * sqrt(4*proximal^2-d^2) + distal*sin(0:0.01:2*pi);
    plot(x_2, y_2,'b')

    % The parametric equation of the sextic S

    phi = 0:0.1:pi;
    ro_1 = 0.5*sqrt(d^2 * cos(2*phi) + ...
        2*d*sin(phi).*sqrt(4*proximal^2-d^2*cos(phi).^2));
    ro_2 = 0.5*sqrt(d^2 * cos(2*phi) - ...
        2*d*sin(phi).*sqrt(4*proximal^2-d^2*cos(phi).^2));

    plot(ro_1.*cos(phi), ro_1.*sin(phi),'r')
    plot(-ro_1.*cos(phi), -ro_1.*sin(phi),'b')
    plot(ro_2.*cos(phi), ro_2.*sin(phi),'r')
    plot(-ro_2.*cos(phi), -ro_2.*sin(phi),'r')
    plot(ro_2.*cos(phi), -ro_2.*sin(phi),'b')
    plot(-ro_2.*cos(phi), ro_2.*sin(phi),'b')
    plot(ro_1.*cos(phi), -ro_1.*sin(phi),'b')
    plot(-ro_1.*cos(phi), ro_1.*sin(phi),'r')
end
%% Plotting bases
plot(-base, 0,'k+')
plot(base, 0,'k+')
title('Task Space')
grid on;
