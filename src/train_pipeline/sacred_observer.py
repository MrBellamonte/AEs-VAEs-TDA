from sacred.observers import RunObserver

class SetID(RunObserver):
    priority = 50  # very high priority

    def __init__(self, custom_id):
        self.custom_id = custom_id

    def started_event(self, ex_info, command, host_info, start_time,  config, meta_info, _id):
        return self.custom_id    # started_event should returns the _id