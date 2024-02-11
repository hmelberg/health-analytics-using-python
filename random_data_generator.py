
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import ast
import random
try:
    from faker import Faker
    __fake=Faker()
except:
    print("Install Faker for full functionality")
import scipy.stats as stats

#%%
def parse_arguments(args_str):
    """>>>parse_arguments("2,color='black')
    [2]{'color':'black'}"""
    args_str = args_str.rstrip(')')  # Remove the closing parenthesis

    args_list = []
    kw_args_dict = {}
    balance = 0
    current_arg = ""
    for char in args_str:
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        if char == ',' and balance == 0:
            # Argument is complete, process it
            if '=' in current_arg:
                key, value = current_arg.split('=', 1)
                kw_args_dict[key.strip()] = ast.literal_eval(value)
            else:
                args_list.append(ast.literal_eval(current_arg))
            current_arg = ""
        else:
            current_arg += char
    # Add the last argument if there is one
    if current_arg:
        if '=' in current_arg:
            key, value = current_arg.split('=', 1)
            kw_args_dict[key.strip()] = ast.literal_eval(value)
        else:
            args_list.append(ast.literal_eval(current_arg))
    print(args_list, kw_args_dict)
    return args_list, kw_args_dict

def random_values_from_distribution(input_str, n):
    # Extract the distribution name and the rest of the string
    dist_name, args_str = input_str.split('(', 1)
    args_str = args_str.rstrip(')')  # Remove the closing parenthesis

    # Use the parse_arguments function to get positional and keyword arguments
    args_list, kw_args_dict = parse_arguments(args_str)

    # Access the distribution function directly from numpy.random using getattr
    try:
        dist_function = getattr(np.random, dist_name)
    except AttributeError:
        raise ValueError(f"Unsupported distribution: {dist_name}")

    # Generate n random samples using the distribution function with both positional and keyword arguments
    return dist_function(*args_list, size=n, **kw_args_dict)


def adjust_probabilities(values):
    """
    Expands a list of values to a new size, maintaining the distribution shape by
    interpolating based on percentage steps in the x-axis.

    Parameters:
    - values: A list of original y-values.
    - new_size: The desired length of the new list.

    Returns:
    - A new list of y-values with length new_size.
    # Example usage
    original_values = np.random.exponential(scale=1.0, size=20)  # Example: an exponential distribution
    original_values.sort()  # Sort to simulate a cumulative shape
    new_values = expand_distribution(original_values, 30)
    """
    new_size=len(values)
    values=[value for value in values if value!=0]
    original_size = len(values)
    if new_size==original_size:
      return values

    # Create an x-axis based on percentage completion through the original list
    original_x = np.linspace(0, 100, original_size)

    # Target x-axis for the new, expanded list
    new_x = np.linspace(0, 100, new_size)

    # Interpolate y-values for the new x-axis
    new_values = np.interp(new_x, original_x, values)

    return new_values

def approximate_probabilities(distribution_name, size=10):
    """
    Generates probabilities for a set number of alternatives such that the probabilities
    approximate the shape of the specified distribution.

    This function uses default parameters for distributions that require them.
    
    #example
    data=approximate_probabilities("poisson", size=7)
    s=pd.Series(data).plot()
    print(data)
    """
    # Sensible defaults for distributions requiring parameters
    default_params = {
        'norm': {'loc': 0, 'scale': 1},  # Mean=0, StdDev=1 for normal
        'poisson': {'mu': 3},  # Lambda (mean) = 3 for Poisson
        'expon': {'scale': 1},  # Mean=1 for exponential
        'lognorm': {'s': 0.954, 'loc': 0, 'scale': 1},
        'beta':   {'a': 2, 'b': 5},  # Symmetric, mildly U-shaped
        'gamma':  {'a': 2, 'scale': 2}  # Moderately right-skewed
    }
    #default_params=get_dynamic_defaults(distribution_name, size)

    params = default_params.get(distribution_name, {})
    #params = default_params

    distribution = getattr(stats, distribution_name)

    if distribution_name in ['uniform']:
        values = 1/size
        probabilities=[values]*size
        print(probabilities)
        return probabilities

    elif distribution_name in ['poisson', 'binom', 'geom']:
        values = np.arange(size)
        densities = distribution.pmf(values, **params)
    else:
        values = np.linspace(-3, 3, size + 1)
        midpoints = (values[:-1] + values[1:]) / 2
        densities = distribution.pdf(midpoints, **params)

    probabilities = densities / densities.sum()
    probabilities = adjust_probabilities(probabilities)

    return probabilities


def calculate_probabilities(expr=None, distribution_name=None, values=None, args=None, kwargs=None):
    """
    Calculate normalized probabilities for given values based on a specified distribution
    and its parameters.

    Parameters:
    - distribution_name (str): The name of the distribution in scipy.stats (e.g., 'norm', 'binom').
    - values (list or np.array): The values/positions for which to calculate probabilities.
    - *args: Positional arguments for the distribution's parameters (e.g., mean and std for 'norm').
    - **kwargs: Keyword arguments for the distribution's parameters.

    Returns:
    - np.array: Normalized probabilities for the given values.
    """

    if expr is not None:
      distribution_name, rest=expr.split("(",1)
      rest = rest.rstrip(")")
      args, kwargs = parse_arguments(rest) if rest else ([], {})
    # Generate default parameters if none are specified
    if not args and not kwargs:
        # Here, instead of using distribution parameters, we directly use generated probabilities
        probabilities = get_default_parameters(distribution_name, size=len(values))
    else:
    # Get the distribution object from scipy.stats
      distribution = getattr(stats, distribution_name)

      # Calculate probability density/mass for each value
      if hasattr(distribution, 'pmf'):
          probabilities = distribution.pmf(values, *args, **kwargs)
      elif hasattr(distribution, 'pdf'):
          probabilities = distribution.pdf(values, *args, **kwargs)
      else:
          raise ValueError(f"The specified distribution '{distribution_name}' does not have a PDF or PMF method.")

    # Normalize the probabilities so they sum to 1
    probabilities /= probabilities.sum()

    return probabilities


def random_values_from_pattern(pattern, numbers=None, letters=None, n=1):
  codes=['']*n
  if letters is None:
    letters = list('abcdefghijklmnopqrstuwxyz')
  if numbers is None:
    numbers = [str(num) for num in range(10)]

  for step, symbol in enumerate(list(pattern)):
    if symbol.isnumeric():
      random_symbol = np.random.choice(numbers, size=n)
      random_symbol = [str(num) for num in random_symbol]

    elif symbol.isalpha():
      if symbol.isupper():
        random_symbol = np.random.choice(letters, size=n)
        random_symbol=[symbol.upper() for symbol in random_symbol]
      else:
        random_symbol = np.random.choice(letters, size=n)

    else:
      random_symbol=[symbol]*n
    codes = list(zip(codes, random_symbol))
    codes = [''.join(code) for code in codes]
  return codes
random_values_from_pattern("K51.7", n=10)

#%%
def random_from_recipe(pattern, n=1):
  """
  pattern='[a-f][1-3][x,y]'
  pattern='[a-f][.,''][1-3][x,y]'
  pattern='[a-f][,;p=(0.5)][1-3][x,y]'
  pattern='[a-f][1-3][x,y][.][1-8]'
  pattern='[a-f,1-3][1-3][x,y][.][1-8]'
  pattern='[a-f][1-3][x,y;p=(0.1, 0.9)][.][1-8]'
  random_code_from_recipe(pattern=pattern, n=10)
  """
  exprs = pattern.split(']')
  exprs = [interval.strip('[') for interval in exprs[:-1]]

  codes=['']*n

  for step, expr in enumerate(exprs):
    hyphen = '-' in expr
    comma = ',' in expr
    longer_than_two = len(expr)>2

    if hyphen & longer_than_two & (not comma):
      if ';p=' in expr:
        expr, p = expr.split(';')
        start, end = expr.split('-')
        start_num = ord(start)
        end_num = ord(end)

        options = [chr(num) for num in range(start_num, end_num)]
        noptions=len(options)
        options.append('')

        p=p.split('(')[1].strip(')')
        p=noptions*[float(p)/noptions]
        p.append(1-sum(p))
        random_symbol = np.random.choice(options, p=p, size=n)
      else:
        #print(expr)
        start, end = expr.split('-')
        start_num = ord(start)
        end_num = ord(end)
        # only works with standard english characters, todo warn if not
        #todo:check if end is inclusive or exclusive, may need to add one, yes think so, but check
        random_int =  [random.randint(start_num, end_num) for num in range(n)]
        random_symbol=[chr(num) for num in random_int]
    elif comma & longer_than_two & (not hyphen):
        if ';p=' in expr:
          expr, p = expr.split(';')
          options = expr.split(',')
          #todo: allow comma and hyphen to be symbols (if escaped?)
          p=p.split('(')[1].strip(')').split(',')
          p=[float(pr) for pr in p]
          if sum(p)<1:
            options.append('')
            p.append(1-sum(p))
          noptions=len(options)
          random_symbol = np.random.choice(options, p=p, size=n)
        else:
          options = expr.split(',')
          #todo: allow comma and hyphen to be symbols (if escaped?)
          noptions=len(options)
          p = noptions * [1/noptions]
          random_symbol = np.random.choice(options, p=p, size=n)

    #  pattern='[a-f,1-3][1-3][x,y][.][1-8]'
    elif comma & longer_than_two & hyphen:
        if ';p=' in expr:
          expr, p = expr.split(';')
          options = expr.split(',')
          p=p.split('(')[1].strip(')').split(',')
          p=[float(pr) for pr in p]
          if sum(p)<1:
            options.append('')
            p.append(1-sum(p))
        else:
          options = expr.split(',')
          #todo: allow comma and hyphen to be symbols (if escaped?)
          noptions=len(options)
          p = noptions * [1/noptions]
        unexpanded_options = options
        expanded_options = []
        expanded_p = []

        for option_num, option in enumerate(unexpanded_options):
          start, end = option.split('-')
          start_num = ord(start)
          end_num = ord(end)
          possible_symbols=[chr(num) for num in range(start_num,end_num+1)]
          expanded_options.extend(possible_symbols)
          n_possible = len(possible_symbols)
          old_p = p[option_num]
          expanded_p.extend(n_possible * [old_p/n_possible])
        random_symbol = np.random.choice(expanded_options, p=expanded_p, size=n)

    else:
      random_symbol=[expr]*n

    codes = list(zip(codes, random_symbol))
    codes = [''.join(symbols) for symbols in codes]

  if len(codes)==1:
    codes=codes[0]
  return codes


def random_items(items=None, probabilities=None, n=1):
    """Selects codes randomly based on provided probabilities using NumPy.
    """

    if isinstance(items, dict):
      codes, probabilities = zip(*items.items())
    if isinstance(probabilities, str):
      probabilities=calculate_probabilities(items, probabilities)

    # Vectorized random selection with replacement
    selected_indices = np.random.choice(len(items), size=n, p=probabilities, replace=True)
    return [items[i] for i in selected_indices]


def make_df(recipe, n):
    """
    specification = dict(
    age="random:normal(50,10)",
    gender={'m':0.51, 'f':0.49},
    icd="pattern:K50.1 distribution:poisson(4)",
    atc=["A004R34", "B004R34"],
    gender={'values':["m", "f"], 'probabilities':[0.51, 0.49]},
    gender={'values':["m", "f"], 'distribution':'uniform(0.5)'},
    visits={'values':[0,1,2,3,4,5], 'prob':'poisson(0.2)'},
    age="if male random:normal(40,5), if female random:normal(50,5)",
    mortality = "table:mortality(age, gender)",
    mortality = age_function,
    mortality = "calc:age*0.1",
    pid = "range(n)",
    dob = "date(01-01-2000, 01-01-2010, stable=pid)")

    specification = dict(
    age="random:normal(50,10)",
    gender={'m':0.51, 'f':0.49},
    icd="pattern:K50.1",
    atc=["A004R34", "B004R34"],
    gender2={'values':["m", "f"], 'probabilities':[0.51, 0.49]},
    gender3={'values':["m", "f"], 'distribution':'uniform()'},
    dob = "date")
    """
    data=pd.DataFrame()


    for k, v in recipe.items():
        print(k, v)
        if callable(v):
            data[k]=data.apply(v)
        elif isinstance(v, list):
            data[k]= np.random.choice(v, size=n, replace=True)
        elif isinstance(v, dict):
            if all(isinstance(vv, float) for kk, vv in v.items()):
                # gender={'m':0.51, 'f':0.49}
                values=list(v.keys())
                prob=list(v.values())
            elif "probabilities" in v:
                # gender={'values':["m", "f"], 'p':[0.51, 0.49]},
                values=v['values']
                prob=v['probabilities']
            elif "distribution" in v:
                # gender={'values':["m", "f"], 'distribution':'uniform(0.5)'},
                values=v['values']
                expr=v['distribution']
                print("expr", expr, values)
                prob=calculate_probabilities(values=values, expr=expr)
                print("prob", prob, values)

            selected_indices = np.random.choice(len(values), size=n, p=prob, replace=True)
            data[k]=[values[i] for i in selected_indices]

        elif isinstance(v, str):
            if v.startswith("random"):
                expr=v.split(":",1)[1]
                data[k]=random_values_from_distribution(expr, n=n)
            elif v.startswith("pattern"):
                expr=v.split(":",1)[1]
                data[k]=random_values_from_pattern(expr, n=n)
            elif v.startswith("df"):
                table, merge_on=v.split("(",1)
                merge_on=merge_on.split()
                data.merge(globals()[table], merge_on=merge_on)
            elif v.startswith("if "):
                expr=v.split(" ",1)
                data[k]=random_values_from_distribution(expr, n=n)
            elif v.startswith("date"):
                if "(" in v:
                    expr, args=v.split("(",1)
                    args, kwargs= parse_arguments(args)
                else:
                    args=[]
                    kwargs={}
                __fake.date(*args, *kwargs)
                data[k]=[__fake.date() for i in range(n)]
        #print(data)
    return data

