#!/usr/bin/perl
#
#  Author: Sara Rosenthal, Preslav Nakov
#  
#  Description: Scores SemEval-2017 task 4, subtask C
#               Using a 5 point scale. Calculates macro-averaged MAE and macro-averaged R for ordinal regression
#
#  Last modified: January 3, 2017
#
# Use:
# (a) outside of CodaLab
#     perl SemEval2017_task4_test_scorer_subtaskC.pl <GOLD_FILE> <INPUT_FILE>
# (b) with CodaLab, i.e., $codalab=1 (certain formatting is expected)
#     perl SemEval2017_task4_test_scorer_subtaskC.pl <INPUT_FILE> <OUTPUT_DIR>
#


use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $GOLD_FILE   =  $ARGV[0];
my $INPUT_FILE  =  $ARGV[1];
my $OUTPUT_FILE =  $INPUT_FILE . '.scored';

my $codalab = 1; # set to 1 if the script is being used in CodaLab


########################
###   MAIN PROGRAM   ###
########################


my %dist = ();
my %count = ();

### 1. Read the files and get the statistics
if ($codalab) {
	my $INPUT_DIR = $ARGV[0];
	print STDERR "Loading input from dir: $INPUT_DIR\n";
 
	opendir(DIR, "$INPUT_DIR/res/") or die $!;

	while (my $file = readdir(DIR)) {

	    # Use a regular expression to ignore files beginning with a period
    	    next if ($file =~ m/^(\.|_)/);
	    $INPUT_FILE = "$INPUT_DIR/res/$file";
	    last;
	}
	closedir(DIR);
	$GOLD_FILE   = "$INPUT_DIR/ref/SemEval2017-task4-dev.subtask-CE.arabic.txt";
	$OUTPUT_FILE = $ARGV[1] . "/scores.txt";
}

print STDERR "Found input file: $INPUT_FILE\n";
open INPUT, $INPUT_FILE or die;

print STDERR "Loading ref data $GOLD_FILE\n";
open GOLD,  $GOLD_FILE or die;

print STDERR "Loading the file...";

for (<INPUT>) {
	s/^[ \t]+//;
	s/[ \n\r]+$//;

	### 1.1. Check the input file format
	die "Wrong file format for $INPUT_FILE: $_" if (!/^(\d+)\t[^\t]+\t(\-?[012])/);
	my ($pid,$proposedLabel) = ($1, $2);

	### 1.2	. Check the gold file format
	$_ = <GOLD>;
	die "Wrong file format!" if (!/^(\d+)\t[^\t]+\t(\-?[012])/);
	my ($tid, $trueLabel) = ($1, $2);

    	die "Ids mismatch!" if ($pid ne $tid);

	### 1.3. Update the statistics
	$dist{$trueLabel} += abs($trueLabel - $proposedLabel);
	$count{$trueLabel}++;
}

while (<GOLD>) {
	die "Missing answer for the following tweet: '$_'\n";
}
print STDERR "DONE\n";

close(INPUT) or die;
close(GOLD) or die;

### 2. Calculate the Scores
print STDERR "Calculating the scores...\n";

print "Creating output file: $OUTPUT_FILE\n";
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;
print STDERR "DONE\n";

my ($absErr, $examplesCnt) = (0, 0);
my $MAE_M = 0.0;
foreach my $class ('-2', '-1', '0', '1', '2') {
    if (defined $count{$class}) {
	my $classDistance = 1.0 * $dist{$class} / $count{$class};
    	$MAE_M += $classDistance;

	$absErr += $dist{$class};
    	$examplesCnt += $count{$class};
	printf OUTPUT "\t%8s: %0.3f\n", $class, $classDistance if (!$codalab);
    }
}
$MAE_M /= 5.0;
my $MAE_mu = 1.0 * $absErr / $examplesCnt;

if ($codalab) {
	printf OUTPUT "MAE_M: %0.3f\nMAE_mu: %0.3f\n", $MAE_M, $MAE_mu;
} else {
	printf OUTPUT "\tOVERALL SCORE : MAE_M=%0.3f, MAE_mu=%0.3f\n", $MAE_M, $MAE_mu;
	print "$INPUT_FILE\t";
	printf "%0.3f\t%0.3f\n", $MAE_M, $MAE_mu;

}

printf STDERR "MAE_M: %0.3f\nMAE_mu: %0.3f\n", $MAE_M, $MAE_mu;
print STDERR "Wrote to $OUTPUT_FILE\n";
close(OUTPUT) or die;


