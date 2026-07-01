// empty header with minimal content, to be used to parse metatomic.h

typedef struct mts_labels_t mts_labels_t;
typedef struct mts_block_t mts_block_t;
typedef struct mts_tensormap_t mts_tensormap_t;

typedef void (*mts_create_array_callback_t)(void*);
typedef void (*mts_realloc_buffer_t)(void*);


typedef struct DLManagedTensorVersioned DLManagedTensorVersioned;
