# Tools
## Summary.py
### Usage

```
Summary.py [-h] [-l WORKDIR_LEVEL] [-p PREFIX] [-o OUTDIR] [-v VAL_LOSS [VAL_LOSS ...]] [-i IGNORE] workdir [workdir ...]

Collect all results of experiments.

positional arguments:
  workdir               the list of output dirs, or the father dir of output dirs

optional arguments:
  -h, --help            show this help message and exit, you can get the new help message by 'python Stat/Summary.py -h'
  
  -l WORKDIR_LEVEL, --workdir_level WORKDIR_LEVEL
                        set the level of workdir, 0: output dirs, 1: the father dir of output dirs
  -p PREFIX, --prefix PREFIX
                        the prefix of the output csv, such as the filename of your output csv is {prefix}_eval_{val_loss}_best_des.csv
  -o OUTDIR, --outdir OUTDIR
                        the output dir of the summary csv
  -v VAL_LOSS [VAL_LOSS ...], --val_loss VAL_LOSS [VAL_LOSS ...]
                        the list of monitor loss, the length of the list should be the same as the length of workdir
  -i IGNORE, --ignore IGNORE
                        ignore the error of no item in evaluation dir, please add this option if your program is not finished
```

## stat_base.py

```
usage: stat_base.py [-h] [-m METRIC] [-n NAME] [-e EPOCH] [-c COLUMN] summary_path

Stat the usefull results of experiments.

positional arguments:
  summary_path          the outdir of Summary.py

optional arguments:
  -h, --help            show this help message and exit
  -m METRIC, --metric METRIC
                        the useful metric to save, use comma to split if you there are multiple metrics
  -n NAME, --name NAME  extra name to save
  -e EPOCH, --epoch EPOCH
                        specify the epoch to save, default is all epochs
  -c COLUMN, --column COLUMN
                        unused
```

## clean_summary

```
usage: clean_summary.py [-h] [-p SUMMARY_PATH] [-n NAME] [-c COLUMNS [COLUMNS ...]] [-nc NEW_COLUMNS [NEW_COLUMNS ...]] [-d DEFAULTS [DEFAULTS ...]] [-r REPLACES [REPLACES ...]]
                        [-t PIVOT_TABLE [PIVOT_TABLE ...]] [--order ORDER [ORDER ...]]

Clean the summary of experiments.

optional arguments:
  -h, --help            show this help message and exit
  -p SUMMARY_PATH, --summary_path SUMMARY_PATH
                        the outdir of Summary.py
  -n NAME, --name NAME  the postfix name of the output file
  -c COLUMNS [COLUMNS ...], --columns COLUMNS [COLUMNS ...]
                        the columns to be output
  -nc NEW_COLUMNS [NEW_COLUMNS ...], --new_columns NEW_COLUMNS [NEW_COLUMNS ...]
                        the new columns name
  -d DEFAULTS [DEFAULTS ...], --defaults DEFAULTS [DEFAULTS ...]
                        the default values for the columns
  -r REPLACES [REPLACES ...], --replaces REPLACES [REPLACES ...]
                        the list of replaces for the columns, $new_column$:old_str:new_str, eg., -r method:tmp:, method:dnn:DNN
  -t PIVOT_TABLE [PIVOT_TABLE ...], --pivot_table PIVOT_TABLE [PIVOT_TABLE ...]
                        the parameters of pivot_table, $value_column$ $columns$, eg., -t Error max_err_limit
  --order ORDER [ORDER ...]
                        sort rows by one column, eg., --order method, you can also specify the order, eg., order method DNN CNN LSTM
```

# Prepare
## Make soft link
To use stat tools for basictorch_v3, you need to soft link the stat tools to your project.
```shell
ln -s $PYTOOLS_PATH/stat_tools Stat
```

Then add 'Stat' to your .gitignore
```
echo -e "\nStat" >> .gitignore
```

# Example

```shell
workdirs='
Gather/tjubd6/Personalized/Compare
'
worklevel=1
summary_path='Summary/Personalized/Compare'
summary_name='all'
metrics='Test Error,Err0'
valid_loss='Valid_Loss'
order='method'
column='+method peoples label_source 200'
new_column='method peoples label_source Error'
replace='method:BaseModel_GT:BaseModel-GT method:Plus:+'

python Stat/Summary.py  $workdirs -o $summary_path -v $valid_loss -l $worklevel
python Stat/stat_base.py $summary_path -m "$metrics"
python Stat/clean_summary.py -p $summary_path/Summary_table.csv -n $summary_name \
                            --order $order -c $column -nc $new_column -r $replace
```