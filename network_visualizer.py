import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras import Model
from keras.models import Sequential

import os

import pickle


class NetworkVisualizer:
    
    def __init__(self, model = None , name = '' , load = False, path_to_directory = None):
        
        # Maybe add a self.check_params(), can be used in load method also and cleaner
        if load:
            
            if path_to_directory is None:
                raise ValueError('To load model please indicate the path to the directory')
                
            self.load( path_to_directory )
        
        elif model is None:
            raise ValueError("Please pass a model")
            
        elif len(model.layers) < 1:
            raise ValueError('model does not contain enough layers')
            
        else:
            
            self.model = model
            self.models_chain = [ ( layer.name , Model( self.model.input , layer.output ) ) for layer in self.model.layers[1:] if layer.name.__contains__('activation') ]
            self.activations = dict()
            self.filtered_activations = dict()
            self.activations_statistics = dict()
            self.concepts_used = set()
            self.layers_monitored = set()
            self.transformed_activations = dict()    
        
    def get_first_layer(self):
        print("function to test class")
        return self.model.layers[0]
    
    def get_activations_generator(self, input_):

        def ret_generator():
            
            ret = None
            
            for name,mod in self.models_chain:
                
                ret = name , mod.predict(input_)

                yield ret

        return ret_generator()
    
    def get_all_activations(self, input_):
        return list( self.get_activations_generator( input_ ) )
    
    def get_activation(self, input_, layer_name):
        
        for name,mod in self.models_chain:
            if layer_name == name:
                return name , mod.predict( input_ )
            
        raise ValueError(f"{layer_name} is not in model.layers")
    
    def update_concept_statistics(self, vect , concept_dico):
    
        vect = vect.flatten()

        for i,value in enumerate( vect ):
            concept_dico[i] = concept_dico.get( i , 0 ) + 1 if value != 0 else concept_dico.get( i , 0 )

        return concept_dico
    
    def update_statistics(self, name, vect, concepts):
        
        dico = self.activations_statistics.get( name )
        
        if dico is None:
            dico = { 'shape' : vect.shape , 'concepts' : dict() , 'n_updates' : 0 }
            
        dico['n_updates'] += 1
            
        for concept in concepts:
            
            dico['concepts'][concept] = self.update_concept_statistics( vect , dico['concepts'].get( concept ,  dict() ) )
            
        self.activations_statistics[name] = dico
            
        
    def update_activations(self, name, vect):
        
        layer_activations = self.activations.get( name , [] )
        layer_activations.append(vect.flatten())
        self.activations[name] = layer_activations
    
    def compute_statistics(self, data, concepts, policy = 'update' , layers_names = None):
        
        if policy not in ['update' , 'overwrite']:
            raise ValueError( "policy can only be 'update' or 'overwrite'" )
            
        if policy == 'overwrite':
            self.activations_statistics = dict()
            self.concepts_used = set()
            self.layers_monitored = set()
            self.activations = dict()
            self.filtered_activations = dict()
            self.transformed_activations = dict()
            
        self.concepts_used.update(concepts)
        
        if layers_names is None:
            self.layers_monitored.update( [ name for name , _ in self.models_chain ] )
        else:
            self.layers_monitored.update( layers_names )
        
        
        for input_ in data:
            activations = self.get_all_activations(input_) if layers_names is None else [ self.get_activation( input_ , name ) for name in layers_names ]
            
            for name, vect in activations:
                
                self.update_statistics( name , vect , concepts )
                self.update_activations( name , vect )
                
    
    def merge_neurons(self, layer_name):
        
        layer_statistics = self.activations_statistics[layer_name]
        
        ret = dict()
        
        for concept in self.concepts_used:
            
            temp_dico = layer_statistics['concepts'][concept]
            
            for key in temp_dico:
                ret[key] = ret.get( key , 0 ) + temp_dico[key]
                
        return ret
    
    
    def filter_activation(self, layer_name, variance_threshold = 0 , min_max_scaler = False, filters = None):
        
        activation_df = pd.DataFrame( self.activations[layer_name] )
        
        if min_max_scaler:
            activation_df = ( activation_df - activation_df.min() ) / ( activation_df.max() - activation_df.min() )
                
        variances = activation_df.var()
        
        filtered = variances[ variances > variance_threshold ]
        
        self.filtered_activations[layer_name] = activation_df[filtered.index.values]
        
        return filtered.index.values
    
    def transform_activation(self, layer_name, transformer):
        
        activation_df = pd.DataFrame( self.activations[layer_name] )
        
        self.transformed_activations[layer_name] = transformer.fit_transform( activation_df )
        
        return self.transformed_activations[layer_name]
    
    def print_layer_statistics(self, layer_name):
        
        layer_statistics = self.activations_statistics[layer_name]
        
        print(f"\t\t\t********** Statistics for layer {layer_name}**********\n")
        
        merged_neurons = self.merge_neurons( layer_name )
        n_unused_neurons = len( [ elem for elem in merged_neurons.values() if elem==0 ] )
        
        print(f"Number of 'unused' neurons (activation == 0) : {n_unused_neurons} out of {np.prod( layer_statistics['shape'] )}")
        print(f"percentage of 'unused' neurons : {(n_unused_neurons / np.prod( layer_statistics['shape'] ))*100} %")
        
        fig = plt.figure( figsize = ( 3 , 2 ) )
        _ = plt.boxplot( list( merged_neurons.values() ) )
        plt.title( f"boxplot for layer {layer_name}" )
        
        plt.show()
        
        print("\n\n\n")
        pass
    
    
    def print_statistics(self, concepts = None , layers_names = None ):
        
        for layer_name in self.layers_monitored:
            self.print_layer_statistics( layer_name )
        pass
            
        
    def save(self, path_to_directory):
        
        keras.models.save_model( self.model ,  os.path.join( path_to_directory , 'model.h5' ) ) 
        
        with open( os.path.join( path_to_directory , 'nkviz.params' ) , 'wb' ) as file:
            
            self.model = None
            self.models_chain = None
            
            pickler = pickle.Pickler(file)
            pickler.dump(self)
            
    def load(self, path_to_directory):
        
        self.model = keras.models.load_model( os.path.join( path_to_directory , 'model.h5' ) )
        self.models_chain = [ ( layer.name , Model( self.model.input , layer.output ) ) for layer in self.model.layers[1:] if layer.name.__contains__('activation') ]
        
        if len(model.layers) < 1:
            raise ValueError('model does not contain enough layers')
            
        with open( os.path.join( path_to_directory , 'nkviz.params' ) , 'rb' ) as file:
            
            nkviz = pickle.Unpickler(file).load()
            
            self.activations_statistics = nkviz.activations_statistics
            self.activations = nkviz.activations
            self.filtered_activations = nkviz.filtered_activations
            self.activations_statistics = nkviz.activations_statistics
            self.concepts_used = nkviz.concepts_used
            self.layers_monitored = nkviz.layers_monitored
            self.transformed_activations = nkviz.transformed_activations   
        