use strict;
use warnings;
use POSIX;

my $nix = 40;
my $niy = 66;
my $tiy = 33;
my $tif = 32;
my $base_addr = 27136000;
my $offset_toy = 6336;
my $offset_of = 12672;

my $nox = 2*$nix;
my $noy = 2*$niy;
my $toy = 2*$tiy;
my $num_tiy = ceil($niy/$tiy);

my $input_words = ceil($nix/32);
my $output_words = ceil($nox/32);

my $input_filename = 'ddr_init.memh';
my $output_filename = 'ddr_expected.memh';
open (RD_FILE, $input_filename)
  or die "";


my @input_rows = <RD_FILE>;
close(RD_FILE);

my $last_input_row = $#input_rows;
my $i = 0;
while ($i <= $last_input_row) {
  my $first_char = substr $input_rows[$i], 0, 1;
  if ($first_char eq "@") {
    splice (@input_rows, $i ,1);
    $last_input_row--;
  } else {
    $i++;
  }
}

open (WR_FILE, ">$output_filename")
  or die "";

my $output_row_cnt = 0;
my $tiy_cnt = 0;
my $tif_cnt = 0;
my $row_cnt = 0;
my $str_cnt = 0;
my $word_cnt = 0;
my $done = 0;

my $cmd_addr = $base_addr;
printf WR_FILE sprintf("\@%08x\n", $cmd_addr);

while ($done == 0) {
  
  my $rd_pntr = $tif_cnt * $niy * $input_words + $tiy_cnt * $tiy * $input_words + $row_cnt * $input_words + $word_cnt;
  my $row = lc $input_rows[$rd_pntr];
  chomp $row;
  
      
  if (length($row) >= 32) {
    my $row_padded = sprintf("%064s", $row);
    
    for my $i (0 .. 1) {
      my $half_row = substr $row_padded, (1-$i)*32, 32;
      my $dup = '';
      my @numbers = ( $half_row =~/../g );
      foreach my $num (@numbers) {
        $dup = join '', $dup , $num, $num;
      } 
      printf WR_FILE $dup;
      printf WR_FILE "\n";
    }
  } else {
    my $row_padded = sprintf("%032s", $row);
    my $first_half_row = substr $row_padded, 0, 32;

    my $dup = '';
    my @numbers = ( $first_half_row =~/../g );
    foreach my $num (@numbers) {
      $dup = join '', $dup , $num, $num;
    }

    printf WR_FILE $dup;
    printf WR_FILE "\n";
  }
  
  $word_cnt++;
  if ($word_cnt == $input_words) {
    $word_cnt = 0;
    $str_cnt++;
    if ($str_cnt == 2) {
      $str_cnt = 0;
      $row_cnt++;
      if ($row_cnt == $tiy) {
        $row_cnt = 0;
        $tif_cnt++;
        if ($tif_cnt == $tif) {
          $tif_cnt = 0;
          $tiy_cnt++;
          if ($tiy_cnt == $num_tiy) {
            $tiy_cnt = 0;
            $done = 1;
          }
        }
        $cmd_addr = $base_addr + $tif_cnt*$offset_of + $tiy_cnt*$offset_toy;
        printf WR_FILE sprintf("\@%08x\n", $cmd_addr);
      }
    }
  }
  $output_row_cnt++;
}

close(WR_FILE);
