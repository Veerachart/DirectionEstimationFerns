/*void fern_based_point_classifier::finalize_training(void)
{

#pragma omp parallel for
  for(int i = 0; i < Ferns->number_of_ferns; i++) {
	  double * number_of_samples_for_this_leaf = new double[Ferns->number_of_leaves_per_fern];
	  memset(number_of_samples_for_this_leaf,0,sizeof(double)*Ferns->number_of_leaves_per_fern);

    double number_of_samples_for_this_fern = 0.;
    for(int j = 0; j < Ferns->number_of_leaves_per_fern; j++)
      for(int k = 0; k < number_of_classes; k++)
	number_of_samples_for_this_fern +=
	  double(leaves_counters[i * step2 + j * step1 + k]) / double(number_of_samples_for_class[k]);

    for(int j = 0; j < Ferns->number_of_leaves_per_fern; j++) {
      for(int k = 0; k < number_of_classes; k++) {
	number_of_samples_for_this_leaf[j] +=
	  double(leaves_counters[i * step2 + j * step1 + k]) / double(number_of_samples_for_class[k]);
      }
    }

    for(int k = 0; k < number_of_classes; k++) {
      double sum = 0.;
      for(int j = 0; j < Ferns->number_of_leaves_per_fern; j++)
	sum += double(leaves_counters[i * step2 + j * step1 + k]) / double(number_of_samples_for_class[k]);

      for(int j = 0; j < Ferns->number_of_leaves_per_fern; j++)
	leaves_distributions[i * step2 + j * step1 + k] =
	  float( log( double(leaves_counters[i * step2 + j * step1 + k]) / double(number_of_samples_for_class[k])
		      / sum
		      / (number_of_samples_for_this_leaf[j] / number_of_samples_for_this_fern) ) );
    }
    delete [] number_of_samples_for_this_leaf;
  }

}


int fern_based_point_classifier::recognize(fine_gaussian_pyramid * pyramid, int u, int v, int level)
{
  int * leaves_index = Ferns->drop(pyramid, u, v, level);

  if (leaves_index == 0) return -1;

  float * distribution = preallocated_distribution_for_a_keypoint;

  for(int i = 0; i < number_of_classes; i++)
    distribution[i] = 0.;

  const int nb_ferns = get_number_of_ferns_to_use();
  for(int i = 0; i < nb_ferns; i++) {
    float * ld = leaves_distributions + i * step2 + leaves_index[i] * step1;
    for(int j = 0; j < number_of_classes; j++)
      distribution[j] += ld[j];
  }

  int class_index = 0;
  float class_score = distribution[0];
  for(int i = 0; i < number_of_classes; i++)
    if (distribution[i] > class_score) {
      class_index = i;
      class_score = distribution[i];
    }

  return class_index;
}
*/
