#!/bin/bash

# Start timing
echo "Script started at: $(date)"

JONTA_ROOT="${PWD}"
PARTICLES_ROOT="${JONTA_ROOT}/particles"

# Args: custom_id ximin ximax  (uses Ebar, alpha, Z_eff, n_steps, dt, N, Energy_threshold from env)
run_moment_pipeline() {
  local custom_id="$1" ximin="$2" ximax="$3" freeze_mode="$4" Energy_threshold="$5"
  cd "$JONTA_ROOT" || exit 1
  # START_TIME=$(date +%s.%N)
  python JaxScript_moments.py \
    --Ebar "$Ebar" --alpha "$alpha" --Z_eff "$Z_eff" \
    --n_steps "$n_steps" --dt "$dt" --N "$N" --run_id "$custom_id" \
    --ximin "$ximin" --ximax "$ximax" --freeze_mode "$freeze_mode"
  cp BinParticles_one_gpu_moments.py "${PARTICLES_ROOT}/${custom_id}"
  cp Plot_bins_moments.py "${PARTICLES_ROOT}/${custom_id}"
  cd "${PARTICLES_ROOT}/${custom_id}" || exit 1
  mkdir -p particles plots data figures
  python BinParticles_one_gpu_moments.py --run_id "$custom_id" --base_dir "${PARTICLES_ROOT}/" \
    --start_time 0 --end_time "$n_steps" --dt 1000 \
    --Ebar "$Ebar" --alpha "$alpha" --Z_eff "$Z_eff" --N "$N"
  python Plot_bins_moments.py --run_id "$custom_id" --base_dir "${PARTICLES_ROOT}/" \
    --start_time 0 --end_time "$n_steps" --dt 1000 --N "$N" \
    --Ebar "$Ebar" --Z_eff "$Z_eff" --alpha "$alpha" \
    --Energy_threshold "$Energy_threshold" 
}

n_steps=20_000
dt=1e-3
N=10_000_000
Energy_threshold=1 #MeV
Ebar=2.5
Z_eff=3.0
alpha=0.1
Ebar=$(printf "%.2f" $Ebar)
Z_eff=$(printf "%.2f" $Z_eff)
alpha=$(printf "%.4f" $alpha)



# ########Predicting moments of runaways#############

####Different initializations
##Aligned with the field
ximin=-1.0
ximax=-0.9
freeze_mode="True"
custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_aligned"
run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"

freeze_mode="False"
custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_aligned_full"
run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"

###Opposed with the field
ximin=0.9
ximax=1.0
freeze_mode="True"
custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_opposed"
run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"

freeze_mode="False"
custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_opposed_full"
run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"
####Scanning Plasma Parameters
ximin=-1.0
ximax=1.0
for Ebar in 4.0 2.5 1.5; do
    Ebar=$(printf "%.2f" $Ebar)
    custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_isotropic"
    freeze_mode="True"
    run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"
    
    custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_isotropic_full"
    freeze_mode="False"
    run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"
done

Ebar=2.5
Ebar=$(printf "%.2f" $Ebar)
alpha=0.1
alpha=$(printf "%.4f" $alpha)
for Z_eff in 1.0 5.0; do
    Z_eff=$(printf "%.2f" $Z_eff)
    custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_isotropic"
    freeze_mode="True"
    run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"

    custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_isotropic_full"
    freeze_mode="False"
    run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"
done

Ebar=2.5
Ebar=$(printf "%.2f" $Ebar)
Z_eff=3.0
Z_eff=$(printf "%.2f" $Z_eff)
for alpha in 0.05 0.2; do
    alpha=$(printf "%.4f" $alpha)
    custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_isotropic"
    freeze_mode="True"
    run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"
    
    custom_id="REmoments_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_isotropic_full"
    freeze_mode="False"
    run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"
done

# ########Predicting moments of runaways#############

# ########Energy distribution#################
ximin=-1.0
ximax=1.0

Ebar=1.5
Z_eff=1.0
alpha=0.2
Ebar=$(printf "%.2f" $Ebar)
Z_eff=$(printf "%.2f" $Z_eff)
alpha=$(printf "%.4f" $alpha)
freeze_mode="False"
custom_id="RE_energy_dist_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_isotropic_full"
run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"

Ebar=2.5
Z_eff=3.0
alpha=0.05
Ebar=$(printf "%.2f" $Ebar)
Z_eff=$(printf "%.2f" $Z_eff)
alpha=$(printf "%.4f" $alpha)
custom_id="RE_energy_dist_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_isotropic_full"
run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"

Ebar=4.0
Z_eff=5.0
alpha=0.1
Ebar=$(printf "%.2f" $Ebar)
Z_eff=$(printf "%.2f" $Z_eff)
alpha=$(printf "%.4f" $alpha)
custom_id="RE_energy_dist_E=${Ebar}_Zeff=${Z_eff}_alpha=${alpha}_isotropic_full"
run_moment_pipeline "$custom_id" "$ximin" "$ximax" "$freeze_mode" "$Energy_threshold"
########Energy distribution#################
