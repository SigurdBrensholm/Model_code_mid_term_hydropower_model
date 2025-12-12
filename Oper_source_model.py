import pyomo.environ as pyo
import numpy as np


### Data ###
# Data is excluded due to confidentiality agreements
# This includes varoiables such as P_e_t, P_c_t, I_t, C_max, C_min, S_max, S_min, eta_plant, V_c_t_max, V_e_t_max, S_0, A_e
# Also other Plant specific data such as turbine and generator data
# Price data is also excluded but can be extracted if plattform access is given
### End of data ###

### Sets ### 
T = list(range(0, 8784))  # Time periods (hours in a leap year)
E = ["DA", "ID Sell", "ID Buy"]  # Energy markets
C = ["aFRR Up", "aFRR Down", "FCR-N", "FCR-D"]  # Capacity markets
C_d = ["aFRR Down", "FCR-N"]  # Subset of C with Downward capacity markets
C_u = ["aFRR Up", "FCR-N", "FCR-D"]  # Subset of C with Upward capacity markets
### End of sets ###

### Functions: ###
# Making coefficient for the constraints:
def create_coeff(x_points, y_points):
    m_list = []
    b_list = []
    for i in range(len(x_points) - 1):
        x1, x2 = x_points[i], x_points[i + 1]
        y1, y2 = y_points[i], y_points[i + 1]
        m = (y2 - y1) / (x2 - x1)
        b = y1
        m_list.append(m)
        b_list.append(b)
    return np.array(m_list), np.array(b_list)

m_turb, b_gen = create_coeff(turbin_list_MW, gen_out)
### End of functions ###



### Model ###
model = pyo.ConcreteModel()

### Sets ###
model.T = pyo.Set(initialize=T, ordered=True)
model.E = pyo.Set(initialize=E)
model.C = pyo.Set(initialize=C)
model.C_d = pyo.Set(initialize=C_d)
model.C_u = pyo.Set(initialize=C_u)

### Parameters ###
model.P_e_t = pyo.Param(model.E, model.T, initialize=P_e_t)
model.P_c_t = pyo.Param(model.C, model.T, initialize=P_c_t)
model.I_t = pyo.Param(model.T, initialize=I_t, mutable = True)
model.C_max = pyo.Param(initialize=C_max)
model.C_min = pyo.Param(initialize=C_min)
model.S_max = pyo.Param(initialize=S_max)
model.S_min = pyo.Param(initialize=S_min)
model.eta_plant = pyo.Param(initialize=eta_plant)
model.V_c_t_max = pyo.Param(model.C, model.T, initialize=V_c_t_max)
model.V_e_t_max = pyo.Param(model.E, model.T, initialize=V_e_t_max)
model.S_0 = pyo.Param(initialize=S_0, mutable=True)
model.A_e = pyo.Param(model.E - {"DA"}, initialize=A_e)  # Only for intraday markets
model.delta_up = pyo.Param(initialize=0.0175)   # Delta energy for up-regulation
model.delta_down = pyo.Param(initialize=0.0095) # Delta down for down-regulation

### Decision variables ###
model.x_e_t = pyo.Var(model.E, model.T, domain=pyo.NonNegativeReals)  # Energy sold/bought in market e at time t [MWh]
model.y_c_t = pyo.Var(model.C, model.T, domain=pyo.NonNegativeReals)  # Capacity sold in market c at time t [MW]
model.s_t = pyo.Var(model.T, within=pyo.NonNegativeReals)  # Reservoir storage level at time t [GWh]
model.P_hyd = pyo.Var(model.T, within=pyo.NonNegativeReals)  # Hydropower drained from reservoiar at time step t [MWh/h]
model.P_shaft = pyo.Var(model.T, within=pyo.NonNegativeReals)  # Power at the turbine shaft [MWh/h]

### Objective Function ###
def objective_function(model):
    return sum(model.P_e_t[e, t] * model.x_e_t[e, t] for e in model.E if e != "ID Buy" for t in model.T) + \
           sum(model.P_c_t[c, t] * model.y_c_t[c, t] for c in model.C for t in model.T) - \
            sum(model.P_e_t["ID Buy", t] * model.x_e_t["ID Buy", t] for t in model.T)
model.OBJ = pyo.Objective(rule=objective_function, sense=pyo.maximize)
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

### Constraints ###
# min and max turbine capacity
def turbine_capacity_rule_max(model, t):
    return sum(model.x_e_t[e, t] for e in model.E if e != "ID Buy")  \
           - model.x_e_t["ID Buy", t] <= model.C_max
model.Turbine_Capacity_Max = pyo.Constraint(model.T, rule=turbine_capacity_rule_max)

def turbine_capacity_rule_min(model, t):
    return sum(model.x_e_t[e, t] for e in model.E if e != "ID Buy") \
           - model.x_e_t["ID Buy", t]>= model.C_min
model.Turbine_Capacity_Min = pyo.Constraint(model.T, rule=turbine_capacity_rule_min)

# min and max reservoir storage
def reservoir_storage_rule_max(model, t):
    return model.s_t[t] <= model.S_max - model.delta_down
model.Reservoir_Storage_Max = pyo.Constraint(model.T, rule=reservoir_storage_rule_max)

def reservoir_storage_rule_min(model, t):
    return model.s_t[t] >= model.S_min + model.delta_up
model.Reservoir_Storage_Min = pyo.Constraint(model.T, rule=reservoir_storage_rule_min)

# Initial reservoir level at t=0
def initial_reservoir_level_rule(model):
    return model.s_t[0] == model.S_0
model.Initial_Reservoir_Level = pyo.Constraint(rule=initial_reservoir_level_rule)

# End reservoiar level at t = T_max
def end_reservoiar_level_rule(model):
    return model.s_t[max(model.T)] == model.S_0
model.End_Reservoiar_Level = pyo.Constraint(rule = end_reservoiar_level_rule)


# Complete turbine and generator PQ constraint using piecewise linear functions:
def pq_constraint_rule(model, t, i):
    return sum(model.x_e_t[e, t] for e in model.E if e != "ID Buy") - model.x_e_t["ID Buy", t] <= m_turb[i]*(model.P_hyd[t] - turbin_list_MW[i]) + b_gen[i]
model.pq_constraint = pyo.Constraint(model.T,range(len(m_turb)), rule=pq_constraint_rule)

# Reservoir Balance Constraint
def reservoir_balance_rule(model, t):
    if t == min(model.T):
        return pyo.Constraint.Skip
    return model.s_t[t] == model.s_t[t-1] + model.I_t[t] - model.P_hyd[t] / 1000
model.Reservoir_Balance = pyo.Constraint(model.T, rule=reservoir_balance_rule)


# Bidding constraint for downwards regulation
def bidding_constraint_down_rule(model, t):
    return (
    sum(model.y_c_t[c, t] for c in model.C_d)
    <=
    sum(model.x_e_t[e, t] for e in model.E if e != "ID Buy")
    - model.x_e_t["ID Buy", t]
    - model.C_min
)
model.Bidding_Constraint_Down = pyo.Constraint(model.T, rule=bidding_constraint_down_rule)

# Bidding constraint for upwards regulation
def bidding_constraint_up_rule(model, t):
    return (
    sum(model.y_c_t[c, t] for c in model.C_u)
    <=
    model.C_max
    - (sum(model.x_e_t[e, t] for e in model.E if e != "ID Buy")
    - model.x_e_t["ID Buy", t])
)
model.Bidding_Constraint_Up = pyo.Constraint(model.T, rule=bidding_constraint_up_rule)

# Capacity market limit
def capacity_market_limit_rule(model, c, t):
    return model.y_c_t[c, t] <= model.V_c_t_max[c, t]
model.Capacity_Market_Limit = pyo.Constraint(model.C, model.T, rule=capacity_market_limit_rule)

# Intraday Buy/Sell market limit
def intraday_market_limit_rule(model, e, t):
    if e == "DA":
        return pyo.Constraint.Skip   # Does not constraint DA
    return model.x_e_t[e, t] <= model.V_e_t_max[e, t]*model.A_e[e]
model.Intraday_Market_Buy_Limit = pyo.Constraint(model.E, model.T, rule=intraday_market_limit_rule)

### Solve the model ###
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)
print("The objective function value:", pyo.value(model.OBJ), "EUR")
print(results.solver.status, results.solver.termination_condition)