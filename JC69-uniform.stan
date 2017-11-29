data {
    int <lower=0> L; // alignment length
    int <lower=0> S; // number of taxa
    real<lower=0,upper=1> tippartials[S,L,4];
    int <lower=0,upper=2*S> peel[S-1,3]; // list of nodes for peeling
}

parameters {
	vector <lower=0,upper=10>[2*S-2] blens; // branch lengths
}

model {
    vector[4] partials[2*S,L];   // partial likelihoods
    matrix[4,4] P[2*S-2]; // probability matrices

    blens ~ exponential(20); // Branch length priors

    // calculate probability matrices
    for( b in 1:2*S-3 ) {
	    for( i in 1:4 ) {
        	for( j in 1:4 ) {
                P[b][i,j] = 0.25 - 0.25*exp(-4*blens[b]/3);
            }
            P[b][i,i] = 0.25 + 0.75*exp(-4*blens[b]/3);
        }
    }
    // zero-length branch length
    P[2*S-2] = diag_matrix(rep_vector(1.0,4));

    // copy tip data into partial likelihoods
    for( n in 1:S ) {
        for( i in 1:L ) {
	        for( j in 1:4 ) {
				partials[n,i][j] = tippartials[n,i,j];
			}
        }
    }
    
    // calculate tree likelihood
	for( i in 1:L ) {
        for( n in 1:(S-1) ) {
            partials[peel[n,3],i] = (P[n*2-1]*partials[peel[n,1],i]) .* (P[n*2]*partials[peel[n,2],i]);
        }
        // root frequencies
        partials[2*S,i] = partials[peel[S-1,3],i] * 0.25;

        // add the site log likelihood
        target += log(sum(partials[2*S,i]));
    }
}

