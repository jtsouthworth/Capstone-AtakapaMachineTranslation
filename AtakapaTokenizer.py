import pandas as pd

class AtakapaTokenizer:
    def __init__(self, lexicon='atakapa_lexicon_final.xlsx', morph_rules='atakapa_morphology_UTF8.xlsx'):
        self.lexicon = pd.read_excel(lexicon)
        self.morph_rules = pd.read_excel(morph_rules)

        clean_apq = self.lexicon['apq'].astype(str).str.strip().tolist()
        norm_roots = set([self._normalize_text(root) for root in clean_apq])
        strip_roots = set([self._remove_accents(root) for root in norm_roots])
        
        self.m_word_exp = sorted([root for root in strip_roots if " " in root], key=len, reverse=True)
        self.valid_roots = {root.replace(' ', '_') for root in strip_roots}
        
        prefix_df = self.morph_rules[self.morph_rules['affix_type'] == 'Prefix']
        self.prefixes = {}
        for _, row in prefix_df.iterrows():
            norm_prefix = self._normalize_text(row['morpheme_action'])
            strip_prefix = self._remove_accents(row['morpheme_action'])
            self.prefixes[strip_prefix] = int(row['slot_position'])
        self.prefix_list = sorted(list(self.prefixes.keys()), key=len, reverse=True)
        
        suffix_df = self.morph_rules[self.morph_rules['affix_type'] == 'Suffix']
        self.suffixes = {}
        for _, row in suffix_df.iterrows():
            norm_suffix = self._normalize_text(row['morpheme_action'])
            strip_suffix = self._remove_accents(row['morpheme_action'])
            self.suffixes[strip_suffix] = int(row['slot_position'])
            
        self.suffix_list = sorted(list(self.suffixes.keys()), key=len, reverse=True)
        
        self.vocab = self._build_vocab()
        self.vocab_inv = {value : key for key, value in self.vocab.items()}

    def _remove_accents(self, text):
        '''Creates a purely unaccented string'''
        text = str(text).lower().strip()
        for p in ['·', 'ˑ', 'ː', '́', '̀']:
            text = text.replace(p, '')
       
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        }
        
        for accented, flat in replacements.items():
            text = text.replace(accented, flat)
            
        return text

    
    def _normalize_text(self, text):
        '''Removes punctuation and applies phonemic representation to phonetic realizations'''
        text = str(text).lower().strip()
        for p in ['.', ',', '?', '!', '…', '"', "'",';']:
            text = text.replace(p, '')
        replacements = {
            'ɬ': 'ƛ', 'ł': 'ƛ'
        }
        for phonetic, phonemic in replacements.items():
            text = text.replace(phonetic, phonemic)
            
        return text

        
    def _build_vocab(self):
        '''Builds vocabulary from roots, prefixes and suffixes'''
        base_dict = {'[PAD]':0 , '[UNK]':1, '[BOS]':2, '[EOS]':3}
        i = 4
        sets = [self.valid_roots, self.prefixes, self.suffixes]
        
        for s in sets:
            for entry in s:
                if entry not in base_dict:
                    base_dict[entry] = i
                    i += 1
        return base_dict


    def _strict_parse(self, current_str, memo=None):
        '''Recursively explores ALL affix paths and returns every valid combination.'''
        if memo is None:
            memo = {}
        if current_str in memo:
            return memo[current_str]
            
        valid_paths = []

        # 1. Base Case
        if current_str in self.valid_roots:
            valid_paths.append([current_str])
                    
        # 2. Suffixes
        for suffix in self.suffixes:
            stripped_suffix = suffix.replace('-', '')
            if current_str.endswith(stripped_suffix) and len(stripped_suffix) > 0:
                sub_paths = self._strict_parse(current_str[:-len(stripped_suffix)], memo)
                for p in sub_paths:
                    valid_paths.append(p + [suffix])
                    
        # 3. Prefixes
        for prefix in self.prefixes:
            stripped_prefix = prefix.replace('-', '')
            if current_str.startswith(stripped_prefix) and len(stripped_prefix) > 0:
                sub_paths = self._strict_parse(current_str[len(stripped_prefix):], memo)
                for p in sub_paths:
                    valid_paths.append([prefix] + p)
                    
        # 4. Unlimited Compounding
        for pivot in range(1, len(current_str)):
            left_root = current_str[:pivot]
            if left_root in self.valid_roots:
                sub_paths = self._strict_parse(current_str[pivot:], memo)
                for p in sub_paths:
                    valid_paths.append([left_root] + p)
                    
        # Save to memo to prevent lag
        memo[current_str] = valid_paths
        return valid_paths


    def _greedy_fallback(self, word):
        '''Peels known affixes off an unknown root as a best-effort fallback.'''
        tokens = []
        current_str = word
        
        # Peel all prefixes
        prefix_found = True
        while prefix_found:
            prefix_found = False
            for prefix in self.prefixes:
                stripped_prefix = prefix.replace('-', '')
                if current_str.startswith(stripped_prefix):
                    prefix_found = True
                    tokens.append(prefix)
                    current_str = current_str[len(stripped_prefix):]
                    break
                    
        # Peel all suffixes
        temp_suffixes = []
        suffix_found = True
        while suffix_found:
            suffix_found = False
            for suffix in self.suffixes:
                stripped_suffix = suffix.replace('-', '')
                if current_str.endswith(stripped_suffix):
                    suffix_found = True
                    temp_suffixes.insert(0, suffix)
                    current_str = current_str[:-len(stripped_suffix)]
                    break
                    
        # Assemble (Prefixes + Unknown Core + Suffixes)
        if current_str:
            tokens.append(current_str)
        tokens.extend(temp_suffixes)
        
        return tokens

    def _align(self, original_word, clean_segments):
        '''Applies the unaccented morphological boundaries back onto accented string'''
        aligned = []
        orig_idx = 0
        removed_chars = set(['·', 'ˑ', 'ː', '́', '̀'])

        for segment in clean_segments:
            clean_str = segment.replace('-', '')
            aligned_segment = ""
            chars_matched = 0
            
            # Rebuild the string using lengths, ignoring 'invisible' marks
            while chars_matched < len(clean_str) and orig_idx < len(original_word):
                c = original_word[orig_idx]
                aligned_segment += c
                orig_idx += 1
                
                if c in removed_chars:
                    pass # Free character, doesn't count against segment length
                else:
                    chars_matched += 1 # Normal or precomposed letter ('á') counts as 1
                    
            # Catch trailing accents/lengths attached to the end of this morpheme
            while orig_idx < len(original_word) and original_word[orig_idx] in removed_chars:
                aligned_segment += original_word[orig_idx]
                orig_idx += 1
                
            # Restore dictionary hyphens 
            if segment.startswith('-') and not aligned_segment.startswith('-'):
                aligned_segment = '-' + aligned_segment
            if segment.endswith('-') and not aligned_segment.endswith('-'):
                aligned_segment = aligned_segment + '-'
                
            aligned.append(aligned_segment)
            
        return aligned


    def _segment_word(self, word):
        """Routes words through parsing logic using the DUAL-LAYER method."""
        # 1. Strip accents for perfect dictionary matching
        clean_word = self._remove_accents(word)
        
        # 2. Parse the clean word
        all_paths = self._strict_parse(clean_word)
        
        best_clean_path = None
        if all_paths:
            grammatical_paths = [p for p in all_paths if self._validate_morphotactics(p)]
            if grammatical_paths:                
                grammatical_paths.sort(key=len)
                best_clean_path = grammatical_paths[0]
                
        if not best_clean_path:
            best_clean_path = self._greedy_fallback(clean_word)
            
        # 3. Align the perfectly parsed clean segments back onto the accented surface word
        return self._align(word, best_clean_path)
        
        
    def _validate_morphotactics(self, path):
        """
        Validates that morphemes follow strict templatic slot order.
        """
        positions = []
        for token in path:
            if token in self.prefixes:
                positions.append(self.prefixes[token])
            elif token in self.suffixes:
                positions.append(self.suffixes[token])
            else:
                positions.append(0) # Roots are 0

        for i in range(len(positions) - 1):
            # 1. Enforce Templatic Slots (Left-to-Right Outward flow)
            if positions[i] > positions[i+1]:
                return False
                
            # 2. Prevent Slot Collisions
            if positions[i] == positions[i+1] and positions[i] != 0:
                return False
              
        return True
   

    def encode(self, text):
        token_ids = []
        token_ids.append(self.vocab['[BOS]'])

        text = text.lower()
        text = self._normalize_text(text)
        
        for mwe in self.m_word_exp:
            if mwe in text:
                text = text.replace(mwe, mwe.replace(' ','_'))
        
        words = text.split()
        for word in words:
            morphemes = self._segment_word(word)
            for morpheme in morphemes:
                morpheme_id = self.vocab.get(morpheme, self.vocab['[UNK]'])
                token_ids.append(morpheme_id)
        
        token_ids.append(self.vocab['[EOS]'])
        return token_ids

    
    def decode(self, token_ids):
        morpheme_strings = []
        ctrl_ids = [0,2,3]
        
        for ids in token_ids:
            if ids in ctrl_ids:
                continue
            elif ids in self.vocab_inv:
                morpheme_strings.append(self.vocab_inv[ids])
        
        final_string = ""
        for m in morpheme_strings:
            m = m.replace('_',' ')

            if final_string == "":
                final_string += m
            elif m.startswith('-'):
                final_string += m.replace('-','')
            elif final_string.endswith('-'):
                final_string = final_string[:-1] + m
            else:
                final_string += " " + m
        
        return final_string