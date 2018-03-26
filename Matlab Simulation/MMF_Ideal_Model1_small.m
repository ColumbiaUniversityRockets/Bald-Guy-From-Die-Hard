%Margaret Mary Master's Thesis
%Ideal Model
%Explicit Method - Forward Difference

clear all
clc

global OUTPUT_RUN_GRAPHS;

RUN_ONCE = 1;
OUTPUT_RUN_GRAPHS = 1;
OUTPUT_GRADIENT_GRAPHS = 1;
OUTPUT_IMPULSE_GRAPH = 1;

OUTPUT_THRUST_CURVE = 1;

MINIMIZE_VALUE = false;     %Minimize or maximize gradient descent value?
%******************************THRUST CURVE DATA************************************%
MOTOR_NAME = "Bald_Guy_From_Die_Hard";
MOTOR_DIAMETER = "98.0";    %Common: 11.0, 13.0, 18.0, 24.0, 29.0, 38.0, 54.0, 
                            %        75.0, 81.0, 98.0, 111.0, 150.0, 161.0
MOTOR_LENGTH = "914.0";
MOTOR_DELAYS = "27";
MOTOR_TOTAL_WEIGHT = "12.5000";
MOTOR_MANUFACTURER = "Elongated_Muskrat";

MOTOR_FILEPATH = "C:\Users\dkola\AppData\Roaming\OpenRocket\ThrustCurves";
MOTOR_FILENAME = "Bald_Guy_From_Die_Hard.eng";
%****************************END THRUST CURVE DATA**********************************%

MOTOR_FP = strcat(MOTOR_FILEPATH,"\",MOTOR_FILENAME);   %Create thrust curve file pointer

global DEnd;
global m_T;
global Fuel_Density;

DEnd = 0.08255;                % Final Port Diameter [m]
m_T = 3.4882;               % tank mass [kg]
Fuel_Density = 812.4;       % [kg/m^3]

global Thrust;

% Given constants

global initImp;
%****************************************************************
global m_loaded;
m_loaded = 4.5;    % N2O mass initially loaded into tank [kg]       //4.3
Main_m_loaded = m_loaded;
%****************************************************************
global Ainj;
Ainj = 0.00002;    % injector area [m^2]                           //0.00191325996 m radius, 1.9133mm
Main_Ainj = Ainj;
%****************************************************************
global V;
V = 0.0081;             % total tank volume [m^3]
Main_V = V;
%****************************************************************
Nozzle_Throat_Diameter = 0.0245;                                       %Diameter of nozzle throat [m]
global A_Star;
A_Star = pi*(Nozzle_Throat_Diameter/2)*(Nozzle_Throat_Diameter/2);      % Nozzle Throat Area [m^3]
Main_A_Star = A_Star;
%****************************************************************
global Length;
Length = 0.25;           % Grain Length [m]
Main_Length = Length;
%****************************************************************
global D0;
D0 = 0.055;              % Initial Port Diameter [m]
Main_D0 = D0;
D = D0;

rawdata = csvread('nestedanal5.csv',1,0);

%The values used in the data from RPA, low, high and step values for OF and
%pressure
OFLow = 1.0;
OFHigh = 35.0;
OFStep = 0.1;

PLow = 101325;
PHigh = 9500000;
PStep = 100000;

global data_temp;
global data_gamma;
global data_Cstar;
global data_thrust;
global data_gridx;
global data_gridy;

      %[    OF        Chamber P    Chamber T      Gamma         C*      Cf]
data = [rawdata(:,1) rawdata(:,2) rawdata(:,6) rawdata(:,8) rawdata(:,10) rawdata(:,13)];
y = PLow:PStep:PHigh;
y = [y PHigh];
%Create the data structures used for the interpolation
[data_gridx,data_gridy] = meshgrid(OFLow:OFStep:OFHigh,y);
data_temp = reshape(data(:,3),[],1 + (OFHigh - OFLow)/OFStep);
data_gamma = reshape(data(:,4),[],1 + (OFHigh - OFLow)/OFStep);
data_Cstar = reshape(data(:,5),[],1 + (OFHigh - OFLow)/OFStep);
data_thrust = reshape(data(:,6),[],1 + (OFHigh - OFLow)/OFStep);

x0 = 0;

tolerance = 1e-12;      %Smallest allowed gradient in gradient descent
maxiter = 1000;         %Maximum # of gradient descent iterations

dxmin = 0.00000001;     %Smallest allowed normalized change in input vector during gradient descent
gnorm = inf; x = x0; niter = 0; dx = inf;

%Step and learning coefficient values for all independent variables
oxStep = 0.004;
oxAlpha = 1e-5;
injStep = 0.0000001;
injAlpha = 1e-14;
grainStep = 0.000001;
grainAlpha = 4e-6;
throatStep = 0.000001;
throatAlpha = 2e-13;
VStep = 0.0000005;
VAlpha = 7e-10;
DStep = 0.000005;
DAlpha = 1e-5;

%Vectors for recording the history of the gradient descent
iter = [0];
oxHist = [m_loaded];
injHist = [Ainj];
grainHist = [Length];
throatHist = [A_Star];
VHist = [V];
DHist = [D0];

%Get the initial impulse
[initImp] = calcImpulse(0,0,0,0,0,0)
impHist = [initImp];

if RUN_ONCE ~= 1
    while and(gnorm >= tolerance, dx >= dxmin)

        %Approximate partial derivitives
        [oxUp] = calcImpulse(oxStep,0,0,0,0,0);
        m_loaded = Main_m_loaded;
        dImpdox = (oxUp - initImp)/oxStep

        [injUp] = calcImpulse(0,injStep,0,0,0,0);
        Ainj = Main_Ainj;
        dImpdinj = (injUp-initImp)/injStep

        [grainUp] = calcImpulse(0,0,grainStep,0,0,0);
        Length = Main_Length;
        dImpdgrain = (grainUp-initImp)/grainStep

        [throatUp] = calcImpulse(0,0,0,throatStep,0,0);
        A_Star = Main_A_Star;
        dImpdthroat = (throatUp-initImp)/throatStep

        [VUp] = calcImpulse(0,0,0,0,VStep,0);
        V = Main_V;
        dImpdV = (VUp-initImp)/VStep

        [DUp] = calcImpulse(0,0,0,0,0,DStep);
        D0 = Main_D0;
        dImpdD0 = (DUp-initImp)/DStep

        %get gnorm to make sure the gradient is large enough
        g = [dImpdox dImpdinj dImpdgrain dImpdthroat dImpdV dImpdD0]
        gnorm = norm(g);
        
        %Use partial derivitives to get new values
        if MINIMIZE_VALUE == false
            oxNew = m_loaded + oxAlpha*dImpdox;
            injNew = Ainj + injAlpha*dImpdinj;
            grainNew = Length + grainAlpha*dImpdgrain;
            throatNew = A_Star + throatAlpha*dImpdthroat;
            VNew = V + VAlpha*dImpdV;
            D0New = D0 + DAlpha*dImpdD0;
        else
            oxNew = m_loaded - oxAlpha*dImpdox;
            injNew = Ainj - injAlpha*dImpdinj;
            grainNew = Length - grainAlpha*dImpdgrain;
            throatNew = A_Star - throatAlpha*dImpdthroat;
            VNew = V - VAlpha*dImpdV;
            D0New = D0 - DAlpha*dImpdD0;
        end

        newVals = [oxNew injNew grainNew throatNew VNew D0New]

        %Get dx to make sure there is a large enough change in the values
        dx = norm([oxNew injNew grainNew throatNew VNew D0New] - [m_loaded Ainj Length A_Star V D0]);

        m_loaded = oxNew;
        Main_m_loaded = oxNew;
        Ainj = injNew;
        Main_Ainj = Ainj;
        Length = grainNew;
        Main_Length = Length;
        A_Star = throatNew;
        Main_A_Star = A_Star;
        V = VNew;
        Main_V = V;
        D0 = D0New;
        Main_D0 = D0;

        %get the new impulse to run through again
        [initImp] = calcImpulse(0,0,0,0,0,0)

        %Record new values in history
        niter = niter+1;
        iter = [iter niter];
        oxHist = [oxHist Main_m_loaded];
        injHist = [injHist Main_Ainj];
        grainHist = [grainHist Main_Length];
        throatHist = [throatHist Main_A_Star];
        VHist = [VHist Main_V];
        DHist = [DHist Main_D0];
        impHist = [impHist initImp];

        %Output new values on graphs
        if OUTPUT_GRADIENT_GRAPHS == 1
            figure(1),plot(iter,oxHist,'r'),title('Oxidizer'),drawnow
            figure(2),plot(iter,injHist,'r'),title('Injector'),drawnow
            figure(3),plot(iter,grainHist,'r'),title('Length'),drawnow
            figure(4),plot(iter,throatHist,'r'),title('A*'),drawnow
            figure(5),plot(iter,VHist,'r'),title('V'),drawnow
            figure(6),plot(iter,DHist,'r'),title('D0'),drawnow
        end
        if OUTPUT_IMPULSE_GRAPH == 1
            figure(7),plot(iter,impHist,'r'),title('impulse'),drawnow
        end
    end
end

%Output thrust curve in .eng format
if OUTPUT_THRUST_CURVE == 1
    %calculate total weight of propellant
    MOTOR_PROP_WEIGHT = num2str(m_loaded + Fuel_Density*pi*realpow((DEnd - D0)/2,2)*Length)
    %Create array of header values
    TC_Arr = [MOTOR_NAME MOTOR_DIAMETER MOTOR_LENGTH MOTOR_DELAYS MOTOR_PROP_WEIGHT ...
        MOTOR_TOTAL_WEIGHT MOTOR_MANUFACTURER];
    %Create array of thrust values
    T = [Thrust(:,1)'; Thrust(:,2)'];
    %Remove first row since this is a 0 row and that signals the end of a
    %.eng file
    T = T(:,2:end);
    TCFile = fopen(MOTOR_FP,'w');
    %output data to .eng file
    fprintf(TCFile,'; Thrust Curve Output by Chuckleton\n');
    fprintf(TCFile,'%s %s %s %s %s %s %s\n',TC_Arr);
    fprintf(TCFile,'   %f %f\n',T);
    fclose(TCFile);
end

%function to calculate the impulse given by the engine given a small change
%to one or more parameters from its original value
function [imp] = calcImpulse(dox,dinj,dgrain,dthroat,dV,dD0)
    Ti = 293.3;              % Test 4
    R = 8314.3;             % universal gas constant [J/(kmol*K)]

    a_coeff = 0.000155;     % aG^n coeffs
    n_coeff = 0.5;

    %parameters of nozzle
    Nozzle_Efficiency = 0.9772;
    Reaction_Efficiency = 0.75;
    Drag_Efficiency = 0.96223;

    %Injector Discharge Coefficient
    Cd = 0.7;

    P0 = 101325;             % Initial Pressure [Pa]
    Pe = P0;                % Chamber Pressure
    
    MW2 = 44.013;           % molecular weight of N2O
    M = 26.2448;            % M of mixture in chamber [g/mol]
    R1 = R / M;             % Adjusted R coefficient
    
    n_He = 0;               % helium gas [kmol]

    
    global m_T;
    global Fuel_Density;
    global initImp;
    global m_loaded;
    m_loaded = m_loaded + dox;
    global Ainj;
    Ainj = Ainj + dinj;
    global V;
    V = V + dV;
    global A_Star;
    A_Star = A_Star + dthroat;
    global Length;
    Length = Length + dgrain;
    global D0;
    D0 = D0 + dD0;
    D = D0;

    %Data used for interpolation
    global data_temp;
    global data_gamma;
    global data_Cstar;
    global data_thrust;
    global data_gridx;
    global data_gridy;
    
    global Thrust;
    
    % Perry's Chemical Engineers' Handbook Property Equations
    G1 = 96.512;        % vapour pressure of N2O [Pa] coefficients
    G2 = -4045;         % valid for Temp range [182.3 K - 309.57 K]
    G3 = -12.277;      
    G4 = 2.886e-5;
    G5 = 2;

    Tc = 309.57;        % critical temperature of N2O [K]
    J1 = 2.3215e7;      % heat of vapourization of N2O [J/kmol] coefficients
    J2 = 0.384;         % valid for Temp range [182.3K - 309.57 K]
    J3 = 0;
    J4 = 0;

    C1 = 0.2079e5;      % heat capacity of He at constant pressure [K/(kmol*K)] coefficients
    C2 = 0;             % valid for Temp range [100 K - 1500 K]
    C3 = 0;
    C4 = 0;
    C5 = 0;

    D1 = 0.2934e5;      % heat capacity of N2O gas at constant pressure [J/(kmol*K)] coefficients
    D2 = 0.3236e5;      % valid for Temp range [100 K - 200 K]
    D3 = 1.1238e3;      
    D4 = 0.2177e5;
    D5 = 479.4;

    E1 = 6.7556e4;      % heat capacity of N2O liquid at constant pressure [J/(kmol*K)] coefficients
    E2 = 5.4373e1;      % valid for Temp range [182.3 K - 200K]
    E3 = 0;
    E4 = 0;
    E5 = 0;

    Q1 = 2.781;         %molar specific volume of liquid N2O [m^3/kmol] coefficients
    Q2 = 0.27244;
    Q3 = 309.57;
    Q4 = 0.2882;

    % Initial conditions
    n_to = m_loaded/MW2;                                        % initial total N2O in tank [kmol]
    Vhat_li = Q2^(1+(1-Ti/Q3)^Q4)/Q1;                           % molar volume of liquid N2O [m^3/kmol]    
    To = Ti;                                                    % initial temperature [K]   
    P_sato = exp(G1 + G2/To + G3*log(To) + G4*To^G5);           % initial vapour pressure of N2O [Pa]
    n_go = P_sato*(V - Vhat_li*n_to)/(-P_sato*Vhat_li + R*To);  % initial N2O gas [kmol]
    n_lo = (n_to*R*To - P_sato*V)/(-P_sato*Vhat_li + R*To);     % initial N2O liquid [kmol]

    % Forward Difference Time Loop
    tf = 9.5;           % final time [s]
    tstep = 0.0005;     % time step [s]
    i_i = 0;
    i_f = tf/tstep;

    for i=i_i:i_f
        t = i*tstep;

        % Given functions of temperature:
        Vhat_l = Q2^(1+(1-To/Q3)^Q4)/Q1;
            %molar specific volume of liquid N2O [m^3/kmol]
        CVhat_He = C1 + C2*To + C3*To^2 + C4*To^3 + C5*To^4 - R;
            %specific heat of He at constant volume [J/(kmol*K)]
        CVhat_g = D1 + D2*((D3/To)/sinh(D3/To))^2 + D4*((D5/To)/cosh(D5/To))^2 - R;
            %specific heat of N2O gas at constant volume [J/(kmol*K)]
        CVhat_l = E1 + E2*To + E3*To^2 + E4*To^3 + E5*To^4;
            %specific heat of N2O liquid at constant volume, approx. same as at
            %constant pressure [J/(kmol*K)]
            Tr = To/Tc;  % reduced temperature
        delta_Hv = J1*(1 - Tr) ^ (J2 + J3*Tr + J4*Tr^2);    % heat of vapourization of N2O [J/kmol]
        P_sat = exp(G1 + G2/To + G3*log(To) + G4*To^G5);    % vapour pressure of N2O [Pa]
        dP_sat = (-G2/(To^2) + G3/To + G4*G5*To^(G5-1)) * exp(G1 + G2/To + G3*log(To) + G4*To^G5);
            %derivative of vapour pressure with respect to temperature
        Cp_T = (4.8 + 0.00322*To)*155.239;                  % specific heat of tank, Aluminum [J/(kg*K)]

        % Simplified expression definitions for solution
        P = (n_He + n_go)*R*To/(V-n_lo*Vhat_l);
        a = m_T*Cp_T + n_He*CVhat_He + n_go*CVhat_g + n_lo*CVhat_l;
        b = P*Vhat_l;
        e = -delta_Hv + R*To;
        f = -Cd*Ainj*sqrt(2/MW2)*sqrt((P-Pe)/Vhat_l);
        j = -Vhat_l*P_sat;
        k = (V - n_lo*Vhat_l)*dP_sat;
        m = R*To;
        q = R*n_go;

        Z = (-f*(-j*a+(q-k)*b))/(a*(m+j) + (q-k)*(e-b));
        W = (-Z*(m*a + (q-k)*e))/(-j*a + (q-k)*b);

        % Derivative Functions
        dT = (b*W+e*Z)/a;
        dn_g = Z;           %Rate (molar) of change of gaseous oxidizer [mol/s]
        dn_l = W;           %Rate (molar) of change of liquid oxidizer [mol/s]

        dn2draindt = -dn_g - dn_l;  %total molar drain rate of oxidizer [mol/s]
        doxdt = dn2draindt * 44.013;    %total drain rate of oxidizer [kg/s]
        
        %Oxidizer cannot return to the tank, should make this throw some
        %kind of error
        if(doxdt < 0) 
            doxdt = 0;
        end

        %regression rate of the fuel grain [m/s] rdot = aG^n, G =
        %doxdt/Aport
        drdt = a_coeff*realpow(doxdt/(pi*D*D/4),n_coeff);
        %Fuel mass flow rate [kg/s]
        dFdt = Fuel_Density * pi * D * Length * drdt;

        %O/F ratio
        of = doxdt/dFdt;

        %interpolated data
        C_Star = interp2(data_gridx,data_gridy,data_Cstar,of,Pe);
        if isnan(C_Star) 
            imp = initImp - 15;
            return
        end
        Thrust_Coeff = interp2(data_gridx,data_gridy,data_thrust,of,Pe);
        Gamma = interp2(data_gridx,data_gridy,data_gamma,of,Pe);
        Temp = interp2(data_gridx,data_gridy,data_temp,of,Pe);
        %Current chamber volume
        Vc = pi*D*D*Length/4;
        %Instantaneous change in chamber pressure
        dPe = (doxdt+dFdt) - Pe*A_Star*sqrt(Gamma/(R1*Temp))...
            *realpow(2/(Gamma+1),(Gamma+1)/(2*(Gamma - 1)));
        dPe = dPe * R1 * Temp / Vc;

        %Thrust [N]
        thrust = (doxdt + dFdt) * C_Star * Nozzle_Efficiency * Reaction_Efficiency * Thrust_Coeff;
        %chamber_pressure = (doxdt + dFdt) * C_Star * Reaction_Efficiency / (A_Star * Drag_Efficiency);
        
        % Record variables for each time step in an array
        T(i+1,1) = t;
        T(i+1,2) = To;
        n_g(i+1,1) = t;
        n_g(i+1,2) = n_go;
        n_l(i+1,1) = t;
        n_l(i+1,2) = n_lo;
        Pres(i+1,1) = t;
        Pres(i+1,2) = P;
        PE(i+1,1) = t;
        PE(i+1,2) = Pe;
        Dn_g(i+1,1) = t;
        Dn_g(i+1,2) = dn_g;
        Dn_l(i+1,1) = t;
        Dn_l(i+1,2) = dn_l;
        Dn_comb(i+1,1) = t;
        Dn_comb(i+1,2) = dn2draindt;
        Dn_ox(i+1,1) = t;
        Dn_ox(i+1,2) = doxdt;
        Dn_F(i+1,1) = t;
        Dn_F(i+1,2) = dFdt;
        Dn_m(i+1,1) = t;
        Dn_m(i+1,2) = doxdt + dFdt;
        Thrust(i+1,1) = t;
        Thrust(i+1,2) = thrust;
        OF(i+1,1) = t;
        OF(i+1,2) = of;
        Diam(i+1,1) = t;
        Diam(i+1,2) = D;
        

        %Foreard Difference Method
        To = To + dT*tstep;
        n_go = n_go + dn_g*tstep;
        n_lo = n_lo + dn_l*tstep;
        Pe = Pe + dPe*tstep;
        D = D + 2*drdt*tstep;

        % Physical stops to kick out of loop
        if Pe>=P        %Chamber pressure cannot be greater than tank pressure (explosion)
            break
        end
        if n_lo<=0      %Stop when no liquid oxidizer remains (gaseous does weird stuff)
            break
        end
    end
    
    %Add in final values
    t = t + tstep;
    
    T(i+1,1) = t;
    T(i+1,2) = To;
    n_g(i+1,1) = t;
    n_g(i+1,2) = n_go;
    n_l(i+1,1) = t;
    n_l(i+1,2) = n_lo;
    Pres(i+1,1) = t;
    Pres(i+1,2) = P;
    PE(i+1,1) = t;
    PE(i+1,2) = Pe;
    Dn_g(i+1,1) = t;
    Dn_g(i+1,2) = 0;
    Dn_l(i+1,1) = t;
    Dn_l(i+1,2) = 0;
    Dn_comb(i+1,1) = t;
    Dn_comb(i+1,2) = 0;
    Dn_ox(i+1,1) = t;
    Dn_ox(i+1,2) = 0;
    Dn_F(i+1,1) = t;
    Dn_F(i+1,2) = 0;
    Dn_m(i+1,1) = t;
    Dn_m(i+1,2) = 0;
    Thrust(i+1,1) = t;
    Thrust(i+1,2) = 0;
    OF(i+1,1) = t;
    OF(i+1,2) = 0;
    Diam(i+1,1) = t;
    Diam(i+1,2) = D;
    
    figure(11), plot(Thrust(:,1),Thrust(:,2), 'r','LineWidth',1),grid, ...
            title('Thrust vs. Time'), ...
            xlabel('Time [s]'), ...
            ylabel('Thrust [N]'),drawnow;

    imp = trapz(Thrust(:,1),Thrust(:,2));
    
    global OUTPUT_RUN_GRAPHS;
    
    if OUTPUT_RUN_GRAPHS == 1
        figure(6), plot(T(:,1),T(:,2), 'r','LineWidth',2),grid, ...
            title('Temperature vs. Time'), ...
            xlabel('Time [s]'), ...
            ylabel('Temperature [K]'),drawnow;
        figure(7), plot(n_g(:,1),n_g(:,2),'b',n_l(:,1),n_l(:,2),'g','LineWidth',2),grid, ...
            title('N2O vs. Time'), ...
            xlabel('Time [s]'), ...
            ylabel('kmol of N2O [kmol]'), ...
            legend('kmol of N2O gas','kmol of N2O liquid'),drawnow;
        figure(8), plot(Pres(:,1),Pres(:,2)*0.000145038,'m',PE(:,1),PE(:,2)*0.000145038,'c','LineWidth',2),grid, ...
            title('Pressure vs. Time'), ...
            xlabel('Time [s]'), ...
            ylabel('Pressure [Psi]'), ...
            legend('tank pressure','chamber pressure'),drawnow; 
        figure(9), plot(Dn_g(:,1),Dn_g(:,2),'m',Dn_l(:,1),Dn_l(:,2),'c',Dn_comb(:,1),Dn_comb(:,2), 'r','LineWidth',2),grid, ...
            title('Molar Flow Rate vs. Time'), ...
            xlabel('Time [s]'), ...
            ylabel('Rate [kmol/s]'), ...
            legend('Gas','Liquid','Combined'),drawnow;
        figure(10), plot(Dn_ox(:,1),Dn_ox(:,2),'m',Dn_F(:,1),Dn_F(:,2),'c',Dn_m(:,1),Dn_m(:,2), 'r','LineWidth',2),grid, ...
            title('Mass Flow Rate vs. Time'), ...
            xlabel('Time [s]'), ...
            ylabel('Rate [kg/s]'), ...
            legend('Oxidizer','Fuel','Combined'),drawnow;
        figure(11), plot(Thrust(:,1),Thrust(:,2), 'r','LineWidth',1),grid, ...
            title('Thrust vs. Time'), ...
            xlabel('Time [s]'), ...
            ylabel('Thrust [N]'),drawnow;
        figure(12), plot(OF(:,1),OF(:,2), 'r','LineWidth',1),grid, ...
            title('OF vs. Time'), ...
            xlabel('Time [s]'), ...
            ylabel('O/F'),drawnow;
        figure(13), plot(Diam(:,1),Diam(:,2), 'r','LineWidth',1),grid, ...
            title('Diameter vs. Time'), ...
            xlabel('Time [s]'), ...
            ylabel('Diameter [m]'),drawnow;
    end
end

%make sure that we don't put too much N2O into the tank (unused atm)
function good = checkICs(m_loaded,Ti,V)
    good = true;
    % Perry's Chemical Engineers' Handbook Property Equations
    G1 = 96.512;        % vapour pressure of N2O [Pa] coefficients
    G2 = -4045;         % valid for Temp range [182.3 K - 309.57 K]
    G3 = -12.277;      
    G4 = 2.886e-5;
    G5 = 2;

    Q1 = 2.781;         %molar specific volume of liquid N2O [m^3/kmol] coefficients
    Q2 = 0.27244;
    Q3 = 309.57;
    Q4 = 0.2882;

    % Initial conditions
    n_to = m_loaded/MW2;                                        % initial total N2O in tank [kmol]
    Vhat_li = Q2^(1+(1-Ti/Q3)^Q4)/Q1;                           % molar volume of liquid N2O [m^3/kmol]    
    To = Ti;                                                    % initial temperature [K]   
    P_sato = exp(G1 + G2/To + G3*log(To) + G4*To^G5);           % initial vapour pressure of N2O [Pa]
    n_go = P_sato*(V - Vhat_li*n_to)/(-P_sato*Vhat_li + R*To);  % initial N2O gas [kmol]
    n_lo = (n_to*R*To - P_sato*V)/(-P_sato*Vhat_li + R*To);     % initial N2O liquid [kmol]
    
    if n_go < (n_to/52.0)
        good = false;
    end
end